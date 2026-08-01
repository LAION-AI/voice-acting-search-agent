#!/usr/bin/env python3
"""Regression test for sidon_enhance fidelity (added after the s0021/s0033 incident:
Sidon's generative decoder can REWRITE heavily degraded input instead of restoring it —
raw corr 0.036, envelope corr 0.56 vs its own input take).

Asserts:
 1. On clean reference speech the restoration IS faithful: envelope corr > 0.85.
 2. envelope_corr discriminates: corr(input, DIFFERENT take) < 0.85.
 3. The tool-level guard rejects unfaithful outputs (min_fidelity honored).

    CUDA_VISIBLE_DEVICES=1 python scripts/test_sidon_fidelity.py [ref_wav]
"""
import os
import sys

import numpy as np
import soundfile as sf

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import tools_ext.sidon as sd


def _read(path):
    w, sr = sf.read(path)
    if getattr(w, "ndim", 1) > 1:
        w = w.mean(1)
    return np.asarray(w, np.float32), sr


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "/tmp/evc_refs/ref1.wav"
    other = "/tmp/evc_refs/ref2.wav"
    w, sr = _read(ref)

    models = sd._load()
    out, osr = sd._enhance(models, w, sr)
    corr = sd.envelope_corr(w, sr, out, osr)
    print(f"[test] clean-file restoration envelope corr = {corr:.3f}")
    assert corr > 0.85, f"FIDELITY REGRESSION: corr {corr:.3f} <= 0.85 on clean input"

    if os.path.exists(other):
        w2, sr2 = _read(other)
        cross = sd.envelope_corr(w, sr, w2, sr2)
        print(f"[test] different-take envelope corr = {cross:.3f}")
        assert cross < 0.85, "envelope_corr does not discriminate different takes"

    # tool-level guard: with an impossible threshold every output must be rejected
    class _Ctx:
        def __init__(self):
            from engine import ToolModelPool
            self.pool = ToolModelPool()
            self.pool._entries["sidon_enhance"] = {
                "obj": models, "vram_gb": 1.5, "ttl_s": 600,
                "last_used": __import__("time").time()}
            self.samples = {"x": {"path": ref, "dur": len(w) / sr, "scores": None}}

        def load_wav(self, sid):
            return w, sr

        def add_sample(self, *a, **k):
            return "new"

    res = sd.run(_Ctx(), ["x"], min_fidelity=1.01)
    assert res["enhanced"][0].get("rejected"), "guard failed to reject below-threshold output"
    print("[test] guard rejection path OK")
    print("[test] ALL SIDON FIDELITY TESTS PASSED")


if __name__ == "__main__":
    main()
