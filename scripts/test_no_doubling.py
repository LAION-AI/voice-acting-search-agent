#!/usr/bin/env python3
"""Regression test for the channel-flatten doubling bug (found 2026-08): the MOSS codec
decode returns stereo, and reshape(-1) CONCATENATED the two identical channels in time,
so every saved sample contained its audio twice back-to-back (half-vs-half envelope corr
0.94-1.0 on all pre-fix artifacts; durations 2x; ASR transcripts contained the text twice).

Asserts, on freshly generated samples:
 1. decode output is channel-averaged, not concatenated: self-repetition envelope corr
    (first half vs second half at +dur/2) < 0.7
 2. duration <= max_frames / 12.5 * 1.05 (codec frame-rate bound)

    CUDA_VISIBLE_DEVICES=1 python scripts/test_no_doubling.py
"""
import os
import sys

import numpy as np

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

FPS = 12.5  # codec frames per second


def self_repetition_corr(w, sr, win=0.05):
    """Envelope corr of first half vs second half (the doubled-audio signature)."""
    n = int(sr * win)
    m = len(w) // n
    env = np.sqrt((w[: m * n].reshape(m, n) ** 2).mean(1))
    h = len(env) // 2
    a, b = env[:h], env[h: 2 * h]
    if h < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(here, "configs/single_gpu.yaml")))
    from engine import Engine
    eng = Engine(cfg)
    max_frames = 200
    outs = eng.generate(
        text="The gates will open at dawn and the caravan leaves soon after, so gather "
             "everything you need and meet me by the northern watchtower before sunrise.",
        instruction="SCRIPT:\n", n=3, max_frames=max_frames, seed=777)
    assert outs, "no samples generated"
    dur_cap = max_frames / FPS * 1.05
    for i, o in enumerate(outs):
        corr = self_repetition_corr(o["wav"], eng.sr)
        print(f"[test] sample {i}: dur={o['dur']}s (cap {dur_cap:.1f}s) "
              f"self-rep corr={corr:.3f}")
        assert corr < 0.7, f"DOUBLING REGRESSION: self-repetition corr {corr:.3f} >= 0.7"
        assert o["dur"] <= dur_cap, f"duration {o['dur']} exceeds codec bound {dur_cap:.1f}"
    print("[test] ALL NO-DOUBLING TESTS PASSED")


if __name__ == "__main__":
    main()
