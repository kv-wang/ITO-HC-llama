#!/usr/bin/env python
"""Test script to verify H_res logging functionality."""

import sys
import torch
from torchtitan.models.llama.model import ModelArgs, Transformer

def test_gamma_collection():
    """Test collecting gamma values from identity_tanh_offdiag model."""
    print("Testing gamma collection...")

    # Create a small model with identity_tanh_offdiag
    model_args = ModelArgs(
        dim=256,
        n_layers=4,
        n_heads=8,
        vocab_size=1000,
        max_seq_len=512,
        hyper_conn_type="identity_tanh_offdiag",
        hyper_conn_n=4,
        mhc_h_res_offdiag_init_scale=0.05,
        mhc_h_res_offdiag_trainable=True,
    )

    with torch.device("meta"):
        model = Transformer(model_args)

    model.to_empty(device="cpu")
    model.init_weights()

    # Collect gamma values
    gamma_dict = {}
    if hasattr(model, 'layers'):
        for layer_id, layer in model.layers.items():
            if hasattr(layer, 'hc_attn') and layer.hc_attn is not None:
                if hasattr(layer.hc_attn, 'h_res_offdiag_log_scale'):
                    gamma_attn = layer.hc_attn.h_res_offdiag_log_scale.exp().item()
                    gamma_dict[f'gamma/layer_{layer_id}_attn'] = gamma_attn
                    print(f"Layer {layer_id} attention gamma: {gamma_attn:.6f}")

            if hasattr(layer, 'hc_ffn') and layer.hc_ffn is not None:
                if hasattr(layer.hc_ffn, 'h_res_offdiag_log_scale'):
                    gamma_ffn = layer.hc_ffn.h_res_offdiag_log_scale.exp().item()
                    gamma_dict[f'gamma/layer_{layer_id}_ffn'] = gamma_ffn
                    print(f"Layer {layer_id} FFN gamma: {gamma_ffn:.6f}")

    print(f"\nCollected {len(gamma_dict)} gamma values")
    return gamma_dict


def test_h_res_extraction():
    """Test extracting H_res matrices."""
    print("\nTesting H_res extraction...")

    model_args = ModelArgs(
        dim=256,
        n_layers=2,
        n_heads=8,
        vocab_size=1000,
        max_seq_len=512,
        hyper_conn_type="identity_tanh_offdiag",
        hyper_conn_n=4,
        mhc_h_res_offdiag_init_scale=0.05,
        mhc_h_res_offdiag_trainable=True,
    )

    with torch.device("meta"):
        model = Transformer(model_args)

    model.to_empty(device="cpu")
    model.init_weights()

    # Extract H_res matrices
    h_res_data = {}
    if hasattr(model, 'layers'):
        for layer_id, layer in model.layers.items():
            if hasattr(layer, 'hc_attn') and layer.hc_attn is not None:
                h_res_attn = layer.hc_attn.h_res_static().detach().cpu().numpy()
                h_res_data[f'layer_{layer_id}_attn'] = h_res_attn
                print(f"Layer {layer_id} attention H_res shape: {h_res_attn.shape}")
                print(f"  Diagonal values: {h_res_attn.diagonal()}")

            if hasattr(layer, 'hc_ffn') and layer.hc_ffn is not None:
                h_res_ffn = layer.hc_ffn.h_res_static().detach().cpu().numpy()
                h_res_data[f'layer_{layer_id}_ffn'] = h_res_ffn
                print(f"Layer {layer_id} FFN H_res shape: {h_res_ffn.shape}")

    print(f"\nExtracted {len(h_res_data)} H_res matrices")
    return h_res_data


if __name__ == "__main__":
    print("=" * 60)
    print("Testing H_res logging functionality")
    print("=" * 60)

    try:
        gamma_dict = test_gamma_collection()
        h_res_data = test_h_res_extraction()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
