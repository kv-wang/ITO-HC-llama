# Hyper-connection residual streams for the Llama model.
#
# Ported from MHC-lite-exploration (hyper_conn/mhc.py and hyper_conn/mhc_lite.py),
# keeping only the pieces needed for these variants (num_fracs = 1, one input view):
#
#   mhc                    H_res = Sinkhorn(static + dynamic logits)            (doubly stochastic)
#   mhc_lite               H_res = sum_r softmax(static + dynamic)_r * P_r      (convex comb. of n! permutations)
#   identity               H_res = I                                            (H_pre / H_post still learned)
#   identity_tanh_offdiag  H_res = I + gamma * offdiag * tanh(static logits)    (static, no dynamic H_res)
#
# Residual layout: (batch, seq, streams, dim). A sublayer f is applied as
#   branch_input = sum_s H_pre[s] * x[s]
#   x'[t]        = sum_s H_res[s, t] * x[s] + H_post[t] * f(branch_input)
#
# The model is built on the meta device and materialized with ``to_empty``, so
# every parameter value is (re)written in ``init_weights``.

import itertools
import math
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.distributed.tensor import distribute_tensor, DTensor
except ImportError:  # older torch
    from torch.distributed._tensor import distribute_tensor, DTensor

HYPER_CONN_TYPES = ("none", "mhc", "mhc_lite", "identity", "identity_tanh_offdiag")
REDUCE_STREAM_MODES = ("sum", "mean", "4mean")
GATE_FNS = ("sigmoid", "softmax")

# submodule names used in TransformerBlock; used to route these params in the optimizer
HYPER_CONN_MODULE_NAMES = ("hc_attn", "hc_ffn")


def is_hyper_conn_param(param_name: str) -> bool:
    return any(f"{name}." in param_name for name in HYPER_CONN_MODULE_NAMES)


def apply_hyper_conn_job_config(model_args, job_config):
    """Copy the --model.hyper_conn_* / --model.mhc_* options into ModelArgs."""
    model_args.hyper_conn_type = job_config.model.hyper_conn_type
    model_args.hyper_conn_n = job_config.model.hyper_conn_n
    model_args.hyper_conn_reduce_mode = job_config.model.hyper_conn_reduce_mode
    model_args.mhc_gate_fn = job_config.model.mhc_gate_fn
    model_args.mhc_sinkhorn_iters = job_config.model.mhc_sinkhorn_iters
    model_args.mhc_h_res_offdiag_init_scale = job_config.model.mhc_h_res_offdiag_init_scale
    model_args.mhc_h_res_offdiag_trainable = not job_config.model.mhc_h_res_offdiag_fixed
    if model_args.hyper_conn_type != "none":
        if job_config.training.tensor_parallel_degree > 1:
            raise NotImplementedError("hyper-connections do not support tensor parallelism yet")
        if job_config.experimental.pipeline_parallel_degree > 1:
            raise NotImplementedError("hyper-connections do not support pipeline parallelism yet")


def hyper_conn_run_tag(model_args) -> str:
    """Suffix for run names / checkpoint folders, empty for the baseline."""
    if model_args.hyper_conn_type == "none":
        return ""
    tag = f"_hc-{model_args.hyper_conn_type}-n{model_args.hyper_conn_n}"
    if model_args.hyper_conn_type == "identity_tanh_offdiag":
        gamma_mode = "trainable" if model_args.mhc_h_res_offdiag_trainable else "fixed"
        tag += f"-gamma{model_args.mhc_h_res_offdiag_init_scale}-{gamma_mode}"
    return tag


def expand_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    """(b, t, d) -> (b, t, s, d), every stream initialized to x ("repeat" mode)."""
    return x.unsqueeze(-2).expand(*x.shape[:-1], num_streams, x.shape[-1]).contiguous()


def reduce_streams(x: torch.Tensor, mode: str) -> torch.Tensor:
    """(b, t, s, d) -> (b, t, d)."""
    if mode == "sum":
        return x.sum(dim=-2)
    if mode == "mean":
        return x.mean(dim=-2)
    if mode == "4mean":
        return x.mean(dim=-2) * 4.0
    raise ValueError(f"Invalid hyper_conn_reduce_mode: {mode}")


def sinkhorn_knopps(log_alpha: torch.Tensor, iters: int = 20) -> torch.Tensor:
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return log_alpha.exp()


_perm_mats_cache: Dict[Tuple[int, str], torch.Tensor] = {}


def permutation_matrices(n: int, device: torch.device) -> torch.Tensor:
    """All n x n permutation matrices, shape (n!, n, n), identity first."""
    key = (n, str(device))
    if key not in _perm_mats_cache:
        index = torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)
        _perm_mats_cache[key] = torch.eye(n, dtype=torch.float32)[index].to(device)
    return _perm_mats_cache[key]


@torch.no_grad()
def _copy_init(param: torch.Tensor, value: torch.Tensor):
    """Copy a full-size init value into a (possibly FSDP-sharded DTensor) tensor."""
    if isinstance(param, DTensor):
        value = value.to(device=param.to_local().device, dtype=param.dtype)
        value = distribute_tensor(value, param.device_mesh, param.placements)
    else:
        value = value.to(device=param.device, dtype=param.dtype)
    param.copy_(value)


class HCRMSNorm(nn.Module):
    """RMSNorm over the flattened (streams * dim) vector, gamma initialized to zero."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

    def init_weights(self):
        _copy_init(self.gamma, torch.zeros(self.gamma.shape))


class HyperConnection(nn.Module):
    def __init__(
        self,
        hyper_conn_type: str,
        num_streams: int,
        dim: int,
        layer_index: int,
        gate_fn: str = "sigmoid",
        sinkhorn_iters: int = 20,
        offdiag_init_scale: float = 0.05,
        offdiag_trainable: bool = True,
    ):
        super().__init__()
        if hyper_conn_type not in HYPER_CONN_TYPES or hyper_conn_type == "none":
            raise ValueError(f"Invalid hyper_conn_type: {hyper_conn_type}")
        if gate_fn not in GATE_FNS:
            raise ValueError(f"Invalid mhc_gate_fn: {gate_fn}")
        if num_streams < 1:
            raise ValueError("hyper_conn_n must be >= 1")
        if offdiag_init_scale <= 0:
            raise ValueError("mhc_h_res_offdiag_init_scale must be > 0")

        n = num_streams
        self.hyper_conn_type = hyper_conn_type
        self.num_streams = n
        self.gate_fn = gate_fn
        self.sinkhorn_iters = sinkhorn_iters
        self.offdiag_init_scale = offdiag_init_scale
        # deterministic (the reference picks a random stream when no layer index is given,
        # which would give different inits on different ranks)
        self.init_residual_index = layer_index % n

        self.norm = HCRMSNorm(n * dim)

        # H_pre / H_post: static logits + scaled dynamic logits
        self.static_alpha_pre = nn.Parameter(torch.empty(n))
        self.pre_branch_scale = nn.Parameter(torch.empty(1))
        self.static_beta = nn.Parameter(torch.empty(n))
        self.h_post_scale = nn.Parameter(torch.empty(1))

        # H_res
        if hyper_conn_type == "mhc":
            self.num_res_logits = n * n
            self.static_alpha_res = nn.Parameter(torch.empty(n, n))
        elif hyper_conn_type == "mhc_lite":
            self.num_res_logits = math.factorial(n)
            self.static_alpha_res = nn.Parameter(torch.empty(self.num_res_logits))
        elif hyper_conn_type == "identity_tanh_offdiag":
            self.num_res_logits = 0  # static only (mhc_disable_dynamic_h_res=True)
            self.static_alpha_res = nn.Parameter(torch.empty(n, n))
            if offdiag_trainable:
                self.h_res_offdiag_log_scale = nn.Parameter(torch.empty(1))
            else:
                self.register_buffer("h_res_offdiag_log_scale", torch.empty(1))
        else:  # identity
            self.num_res_logits = 0
        if self.num_res_logits > 0:
            self.residual_scale = nn.Parameter(torch.empty(1))

        # one projection for all dynamic logits, columns = [pre (n) | post (n) | res]
        self.dynamic_fn = nn.Parameter(torch.empty(n * dim, 2 * n + self.num_res_logits))

    def init_weights(self):
        n = self.num_streams
        self.norm.init_weights()

        one_hot_logits = torch.full((n,), -1.0)
        one_hot_logits[self.init_residual_index] = 1.0
        _copy_init(self.static_alpha_pre, one_hot_logits)
        _copy_init(self.static_beta, one_hot_logits)
        _copy_init(self.pre_branch_scale, torch.full((1,), 1e-2))
        _copy_init(self.h_post_scale, torch.full((1,), 1e-2))
        _copy_init(self.dynamic_fn, torch.zeros(self.dynamic_fn.shape))

        if self.hyper_conn_type == "mhc":
            # near-identity logits: 0 on the diagonal, -8 elsewhere
            res = torch.full((n, n), -8.0)
            res.fill_diagonal_(0.0)
            _copy_init(self.static_alpha_res, res)
        elif self.hyper_conn_type == "mhc_lite":
            # almost all mass on the identity permutation (index 0)
            res = torch.full((self.num_res_logits,), -8.0)
            res[0] = 0.0
            _copy_init(self.static_alpha_res, res)
        elif self.hyper_conn_type == "identity_tanh_offdiag":
            _copy_init(self.static_alpha_res, torch.zeros(n, n))
            _copy_init(
                self.h_res_offdiag_log_scale,
                torch.full((1,), math.log(self.offdiag_init_scale)),
            )
        if self.num_res_logits > 0:
            _copy_init(self.residual_scale, torch.full((1,), 1e-2))

    def _gate(self, logits: torch.Tensor, scale: float) -> torch.Tensor:
        if self.gate_fn == "softmax":
            return F.softmax(logits, dim=-1)
        return logits.sigmoid() * scale

    def h_res_static(self) -> torch.Tensor:
        """identity_tanh_offdiag: I + gamma * offdiag * tanh(logits)."""
        logits = self.static_alpha_res.float()
        eye = torch.eye(self.num_streams, device=logits.device, dtype=logits.dtype)
        gamma = self.h_res_offdiag_log_scale.float().exp()
        return eye + gamma * (1.0 - eye) * logits.tanh()

    def forward(
        self, residuals: torch.Tensor
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """
        Args:
            residuals: (b, t, s, d) residual streams.
        Returns:
            branch_input (b, t, d) and a function mapping the branch output (b, t, d)
            to the updated residual streams (b, t, s, d).
        """
        n = self.num_streams
        dtype = residuals.dtype

        # coefficients are computed in fp32 (sinkhorn / softmax under bf16 params is fragile)
        normed = self.norm(residuals.flatten(-2))
        dynamic = (normed @ self.dynamic_fn).float()
        dyn_pre, dyn_post, dyn_res = dynamic.split([n, n, self.num_res_logits], dim=-1)

        h_pre = self._gate(
            self.static_alpha_pre.float() + self.pre_branch_scale.float() * dyn_pre, 1.0
        )
        h_post = self._gate(
            self.static_beta.float() + self.h_post_scale.float() * dyn_post, 2.0
        )

        branch_input = torch.einsum("...s,...sd->...d", h_pre.to(dtype), residuals)

        if self.hyper_conn_type == "identity":
            mixed = residuals
        elif self.hyper_conn_type == "identity_tanh_offdiag":
            h_res = self.h_res_static().to(dtype)
            mixed = torch.einsum("st,...sd->...td", h_res, residuals)
        else:
            if self.hyper_conn_type == "mhc":
                # dynamic columns are laid out row-major as (s t)
                dyn_res = dyn_res.unflatten(-1, (n, n))
            res_logits = self.static_alpha_res.float() + self.residual_scale.float() * dyn_res
            if self.hyper_conn_type == "mhc":
                h_res = sinkhorn_knopps(res_logits, self.sinkhorn_iters)
            else:  # mhc_lite
                perms = permutation_matrices(n, res_logits.device)
                h_res = torch.einsum("...r,rij->...ij", res_logits.softmax(dim=-1), perms)
            mixed = torch.einsum("...st,...sd->...td", h_res.to(dtype), residuals)

        h_post = h_post.to(dtype).unsqueeze(-1)

        def add_residual(branch_output: torch.Tensor) -> torch.Tensor:
            return mixed + h_post * branch_output.unsqueeze(-2)

        return branch_input, add_residual
