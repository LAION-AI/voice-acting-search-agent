"""audio_stretch — time-stretch WITHOUT pitch change via audiostretchy (CPU).

ratio 0.25-4.0: >1.0 = slower/longer, <1.0 = faster/shorter. New samples keep the
source genome metadata; stretched audio must be re-scored.
"""
import os
import tempfile

import numpy as np
import soundfile as sf

VRAM_GB = 0.0  # pure CPU


def run(ctx, sample_ids, ratio):
    ratio = float(ratio)
    if not 0.25 <= ratio <= 4.0:
        return {"error": f"ratio {ratio} outside [0.25, 4.0]"}
    from audiostretchy.stretch import stretch_audio
    out = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        src = ctx.samples[sid]["path"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp = t.name
        try:
            stretch_audio(src, tmp, ratio=ratio)
            wav, sr = sf.read(tmp)
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(1)
            wav = np.asarray(wav, np.float32)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        meta = {k: v for k, v in ctx.samples[sid].items()
                if k not in ("path", "scores")}
        meta.update({"stretched_from": sid, "ratio": ratio, "tool": "audio_stretch"})
        nid = ctx.add_sample(wav, round(len(wav) / sr, 2), meta, sr=sr)
        out.append({"src": sid, "new_sample_id": nid,
                    "dur": round(len(wav) / sr, 2)})
    return {"stretched": out, "ratio": ratio,
            "note": "stretched samples are NEW samples; re-score them"}
