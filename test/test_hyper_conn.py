import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.llama.hyper_conn import (
    HyperConnection,
    is_hyper_conn_param,
    permutation_matrices,
    sinkhorn_knopps,
)
from torchtitan.models.llama.model import ModelArgs, Transformer

HC_TYPES = ("mhc", "mhc_lite", "identity", "identity_tanh_offdiag")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def small_args(hyper_conn_type, **kwargs):
    return ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        max_seq_len=32,
        hyper_conn_type=hyper_conn_type,
        **kwargs,
    )


def build_model(hyper_conn_type, seed=0, **kwargs):
    torch.manual_seed(seed)
    return Transformer(small_args(hyper_conn_type, **kwargs)).to(DEVICE)


def perturb_(module, std=0.5, seed=1):
    """Move HC params away from their init so H matrices are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * std)


class TestHResConstraints(unittest.TestCase):
    def test_sinkhorn_doubly_stochastic(self):
        logits = torch.randn(5, 4, 4)
        h = sinkhorn_knopps(logits, 20)
        # the last sinkhorn step normalizes rows exactly; columns are approximate
        torch.testing.assert_close(h.sum(-1), torch.ones(5, 4), atol=1e-5, rtol=0)
        torch.testing.assert_close(h.sum(-2), torch.ones(5, 4), atol=1e-2, rtol=0)
        self.assertTrue((h >= 0).all())

    def test_permutation_matrices(self):
        perms = permutation_matrices(4, torch.device("cpu"))
        self.assertEqual(perms.shape, (24, 4, 4))
        torch.testing.assert_close(perms[0], torch.eye(4))
        torch.testing.assert_close(perms.sum(-1), torch.ones(24, 4))
        torch.testing.assert_close(perms.sum(-2), torch.ones(24, 4))
        self.assertEqual(len({tuple(p.flatten().tolist()) for p in perms}), 24)

    def _hc(self, hc_type, **kwargs):
        torch.manual_seed(0)
        hc = HyperConnection(hc_type, num_streams=4, dim=16, layer_index=1, **kwargs)
        hc.init_weights()
        return hc

    def _h_res_matrix(self, hc):
        """Per-token H_res, shape (b, t, n, n), by feeding one-hot residual streams."""
        n = hc.num_streams
        x = torch.randn(2, 3, n, 16)
        # recompute coefficients exactly as forward does
        normed = hc.norm(x.flatten(-2))
        dyn = (normed @ hc.dynamic_fn).float()
        _, _, dyn_res = dyn.split([n, n, hc.num_res_logits], dim=-1)
        if hc.hyper_conn_type == "mhc":
            logits = hc.static_alpha_res + hc.residual_scale * dyn_res.unflatten(-1, (n, n))
            h = sinkhorn_knopps(logits, hc.sinkhorn_iters)
        elif hc.hyper_conn_type == "mhc_lite":
            logits = hc.static_alpha_res + hc.residual_scale * dyn_res
            h = torch.einsum("...r,rij->...ij", logits.softmax(-1), permutation_matrices(n, x.device))
        elif hc.hyper_conn_type == "identity_tanh_offdiag":
            h = hc.h_res_static().expand(2, 3, n, n)
        else:
            h = torch.eye(n).expand(2, 3, n, n)
        # check that forward mixes the streams with exactly this matrix
        _, add_residual = hc(x)
        mixed = add_residual(torch.zeros(2, 3, 16))
        torch.testing.assert_close(mixed, torch.einsum("btsu,btsd->btud", h, x), atol=1e-5, rtol=1e-5)
        return h

    def test_init_is_near_identity(self):
        for hc_type in HC_TYPES:
            with self.subTest(hc_type=hc_type):
                h = self._h_res_matrix(self._hc(hc_type))
                # off-identity logits start at -8: mhc_lite keeps ~23 * e^-8 of the mass off I
                torch.testing.assert_close(h, torch.eye(4).expand_as(h), atol=1e-2, rtol=0)

    def test_constraints_after_training_like_perturbation(self):
        for hc_type in ("mhc", "mhc_lite"):
            with self.subTest(hc_type=hc_type):
                hc = self._hc(hc_type)
                perturb_(hc)
                h = self._h_res_matrix(hc)
                self.assertTrue((h >= -1e-6).all())
                torch.testing.assert_close(h.sum(-1), torch.ones(2, 3, 4), atol=1e-5, rtol=0)
                # mhc: 20 sinkhorn iterations (as in the reference) leave columns approximate
                col_atol = 5e-2 if hc_type == "mhc" else 1e-5
                torch.testing.assert_close(h.sum(-2), torch.ones(2, 3, 4), atol=col_atol, rtol=0)

        hc = self._hc("identity_tanh_offdiag")
        perturb_(hc, std=3.0)
        h = self._h_res_matrix(hc)[0, 0]
        gamma = hc.h_res_offdiag_log_scale.exp().item()
        torch.testing.assert_close(torch.diagonal(h), torch.ones(4))
        offdiag = h[~torch.eye(4, dtype=torch.bool)]
        self.assertTrue((offdiag.abs() <= gamma + 1e-6).all())
        self.assertTrue((offdiag.abs() > 0).any())

        hc = self._hc("identity")
        perturb_(hc)
        h = self._h_res_matrix(hc)
        torch.testing.assert_close(h, torch.eye(4).expand_as(h))

    def test_offdiag_gamma_trainable_flag(self):
        hc = self._hc("identity_tanh_offdiag", offdiag_init_scale=0.05)
        self.assertIn("h_res_offdiag_log_scale", dict(hc.named_parameters()))
        self.assertAlmostEqual(hc.h_res_offdiag_log_scale.exp().item(), 0.05, places=6)
        hc = self._hc("identity_tanh_offdiag", offdiag_trainable=False)
        self.assertNotIn("h_res_offdiag_log_scale", dict(hc.named_parameters()))
        self.assertIn("h_res_offdiag_log_scale", dict(hc.named_buffers()))

    def test_pre_post_init(self):
        hc = self._hc("mhc")  # layer_index=1 -> stream 1
        x = torch.randn(2, 3, 4, 16)
        branch_input, add_residual = hc(x)
        pre = torch.sigmoid(torch.tensor([-1.0, 1.0, -1.0, -1.0]))
        torch.testing.assert_close(branch_input, torch.einsum("s,btsd->btd", pre, x), atol=1e-5, rtol=1e-5)
        out = torch.randn(2, 3, 16)
        delta = add_residual(out) - add_residual(torch.zeros_like(out))
        torch.testing.assert_close(delta, 2 * pre.view(1, 1, 4, 1) * out.unsqueeze(-2), atol=1e-5, rtol=1e-5)


class TestTransformerWithHyperConn(unittest.TestCase):
    def test_baseline_has_no_hc_params(self):
        model = build_model("none")
        self.assertFalse(any(is_hyper_conn_param(n) for n, _ in model.named_parameters()))
        self.assertTrue(all(layer.hc_attn is None for layer in model.layers.values()))

    def test_meta_init_matches_eager(self):
        for hc_type in HC_TYPES:
            with self.subTest(hc_type=hc_type):
                eager = build_model(hc_type)
                with torch.device("meta"):
                    meta = Transformer(small_args(hc_type))
                meta.to_empty(device=DEVICE)
                meta.init_weights()
                eager_sd, meta_sd = eager.state_dict(), meta.state_dict()
                self.assertEqual(eager_sd.keys(), meta_sd.keys())
                hc_keys = [k for k in eager_sd if is_hyper_conn_param(k)]
                self.assertTrue(hc_keys)
                for k in hc_keys:
                    torch.testing.assert_close(meta_sd[k], eager_sd[k], msg=k)

    def test_forward_backward(self):
        tokens = torch.randint(0, 128, (2, 32), device=DEVICE)
        for hc_type in HC_TYPES:
            with self.subTest(hc_type=hc_type):
                model = build_model(hc_type)
                logits = model(tokens)
                self.assertEqual(logits.shape, (2, 32, 128))
                loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                for layer in model.layers.values():
                    self.assertGreater(layer.hc_attn.dynamic_fn.grad.abs().sum().item(), 0)
                    self.assertGreater(layer.hc_attn.static_alpha_pre.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "fused_rmsnorm needs CUDA")
    def test_fused_rmsnorm_matches_rmsnorm(self):
        # the final norm sees non-contiguous HC streams (einsum outputs) and feeds the
        # stream reduction, whose grad is non-contiguous
        tokens = torch.randint(0, 128, (2, 32), device=DEVICE)
        for hc_type in HC_TYPES:
            with self.subTest(hc_type=hc_type):
                logits, grads = {}, {}
                for norm_type in ("rmsnorm", "fused_rmsnorm"):
                    model = build_model(hc_type, norm_type=norm_type)
                    logits[norm_type] = model(tokens)
                    F.cross_entropy(logits[norm_type].flatten(0, 1), tokens.flatten()).backward()
                    grads[norm_type] = {n: p.grad for n, p in model.named_parameters()}
                torch.testing.assert_close(logits["fused_rmsnorm"], logits["rmsnorm"], atol=1e-4, rtol=1e-3)
                for name, g in grads["rmsnorm"].items():
                    torch.testing.assert_close(grads["fused_rmsnorm"][name], g, atol=1e-4, rtol=1e-3, msg=name)

    def test_single_stream_runs(self):
        # with one stream, H_pre / H_post reduce to per-token scalar gates
        model = build_model("identity", hyper_conn_n=1)
        tokens = torch.randint(0, 128, (2, 32), device=DEVICE)
        self.assertEqual(model(tokens).shape, (2, 32, 128))

    def test_adam_mini_training_reduces_loss(self):
        try:
            from adam_mini import Adam_mini
        except ImportError:
            self.skipTest("adam_mini not installed")
        torch.manual_seed(0)
        tokens = torch.randint(0, 128, (4, 33), device=DEVICE)
        inputs, labels = tokens[:, :-1], tokens[:, 1:]
        for hc_type in HC_TYPES:
            with self.subTest(hc_type=hc_type):
                model = build_model(hc_type)
                args = model.model_args
                opt = Adam_mini(
                    named_parameters=model.named_parameters(),
                    lr=3e-3,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    verbose=False,
                )
                # same routing as train.py
                opt.adam_block_names.update(("hc_attn", "hc_ffn"))
                for group in opt.param_groups:
                    if is_hyper_conn_param(group["name"]):
                        group["weight_decay"] = 0.0
                losses = []
                for _ in range(30):
                    opt.zero_grad()
                    loss = F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                self.assertLess(losses[-1], losses[0] - 1.0, losses)
                # HC params were updated by per-coordinate Adam
                hc = model.layers["0"].hc_attn
                self.assertIn("v", opt.state[hc.dynamic_fn])
                self.assertGreater(hc.dynamic_fn.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
