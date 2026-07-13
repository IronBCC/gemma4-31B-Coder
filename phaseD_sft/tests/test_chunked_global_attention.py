import math
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # local data-pipeline venv has no torch; run on the training box
    raise unittest.SkipTest("torch not installed in this environment")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseD_sft.chunked_global_attention import chunked_causal_attention


def dense_causal_reference(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    s = scores.shape[-1]
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class ChunkedCausalAttentionTests(unittest.TestCase):
    def _compare(self, B, H, S, D, q_block, dtype=torch.float32, atol=1e-5, rtol=1e-4):
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
        q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

        ref = dense_causal_reference(q, k, v, scale)
        got = chunked_causal_attention(q2, k2, v2, scale=scale, q_block=q_block)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)
        got.backward(grad_out)
        torch.testing.assert_close(q2.grad, q.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(k2.grad, k.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(v2.grad, v.grad, atol=atol, rtol=rtol)

    def test_exact_small_block_divides(self):
        self._compare(B=1, H=2, S=64, D=16, q_block=16)

    def test_exact_jagged_block(self):
        # S not divisible by q_block — tail block path
        self._compare(B=1, H=2, S=77, D=16, q_block=32)

    def test_exact_single_block(self):
        # q_block >= S degenerates to dense-in-one-block
        self._compare(B=1, H=1, S=48, D=8, q_block=64)

    def test_exact_head_dim_512_like(self):
        self._compare(B=1, H=2, S=96, D=512, q_block=32, atol=5e-4, rtol=1e-3)

    def test_scale_none_defaults_to_rsqrt_d(self):
        torch.manual_seed(1)
        q = torch.randn(1, 1, 32, 8)
        k, v = torch.randn_like(q), torch.randn_like(q)
        got = chunked_causal_attention(q.clone(), k.clone(), v.clone(), scale=None, q_block=8)
        ref = dense_causal_reference(q, k, v, 8 ** -0.5)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)

    def test_bf16_matches_fp32_reference_loosely(self):
        torch.manual_seed(2)
        B, H, S, D = 1, 2, 64, 32
        scale = 1.0 / math.sqrt(D)
        qf = torch.randn(B, H, S, D)
        kf, vf = torch.randn_like(qf), torch.randn_like(qf)
        ref = dense_causal_reference(qf, kf, vf, scale)
        got = chunked_causal_attention(
            qf.to(torch.bfloat16), kf.to(torch.bfloat16), vf.to(torch.bfloat16),
            scale=scale, q_block=16,
        )
        torch.testing.assert_close(got.float(), ref, atol=3e-2, rtol=3e-2)

    def test_causality_first_token_ignores_future(self):
        torch.manual_seed(3)
        q = torch.randn(1, 1, 16, 8)
        k, v = torch.randn_like(q), torch.randn_like(q)
        out1 = chunked_causal_attention(q.clone(), k.clone(), v.clone(), q_block=4)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 8:], v2[:, :, 8:] = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
        out2 = chunked_causal_attention(q.clone(), k2, v2, q_block=4)
        torch.testing.assert_close(out1[:, :, :8], out2[:, :, :8], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
