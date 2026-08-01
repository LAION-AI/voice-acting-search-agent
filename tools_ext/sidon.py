"""sidon_enhance — speech restoration via Sidon v0.1 (sarulab-speech/sidon-v0.1).

TorchScript feature extractor + decoder over SeamlessM4T (w2v-BERT-2.0) features;
input any sr, output 48 kHz. Lazy-loaded through ctx.pool; never loaded at startup.
"""
import numpy as np
import torch

VRAM_GB = 1.5
_DEVICE = "cuda"


def _load():
    import transformers
    from huggingface_hub import hf_hub_download
    fe_path = hf_hub_download("sarulab-speech/sidon-v0.1",
                              filename="feature_extractor_cuda.pt")
    dec_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")
    fe = torch.jit.load(fe_path, map_location=_DEVICE).to(_DEVICE)
    decoder = torch.jit.load(dec_path, map_location=_DEVICE).to(_DEVICE)
    pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0", sampling_rate=16000)
    return {"fe": fe, "decoder": decoder, "pre": pre}


def _enhance(models, wav, sr):
    """1-D float32 wav @ sr -> restored 1-D float32 wav @ 48kHz (reference pipeline)."""
    import torchaudio
    waveform = torch.as_tensor(np.asarray(wav, np.float32)).reshape(1, -1)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = 0.9 * (waveform / peak)
    target_n = int(48_000 / sr * waveform.shape[-1])
    w16 = torchaudio.functional.highpass_biquad(waveform, sr, 50)
    w16 = torchaudio.functional.resample(w16, sr, 16_000)
    w16 = torch.nn.functional.pad(w16, (0, 24000))
    restoreds = []
    feature_cache = None
    for chunk in w16.view(-1).split(16000 * 96):
        inputs = models["pre"](torch.nn.functional.pad(chunk, (160, 160)),
                               return_tensors="pt")
        with torch.inference_mode():
            feature = models["fe"](inputs["input_features"].to(_DEVICE))["last_hidden_state"]
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
            restoreds.append(models["decoder"](feature.transpose(1, 2)).view(-1)[:-960])
            feature_cache = feature[:, -1:]
    out = torch.cat(restoreds, dim=0)[:target_n]
    return out.float().cpu().numpy(), 48000


def envelope_corr(wav_a, sr_a, wav_b, sr_b, win_s=0.05):
    """Fidelity metric: Pearson corr of 50ms RMS envelopes (both resampled to 16k).
    A faithful restoration of the SAME take scores >0.9; generative divergence scores low."""
    import torchaudio

    def env(w, sr):
        t = torch.as_tensor(np.asarray(w, np.float32))
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        w2 = t.numpy()
        n = int(16000 * win_s)
        m = len(w2) // n
        return np.sqrt((w2[: m * n].reshape(m, n) ** 2).mean(1))

    ea, eb = env(wav_a, sr_a), env(wav_b, sr_b)
    m = min(len(ea), len(eb))
    ea, eb = ea[:m], eb[:m]
    if m < 10 or ea.std() < 1e-9 or eb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


# Below this envelope correlation the output is NOT a faithful restoration of the input
# take (Sidon's generative decoder rewrites heavily degraded audio) -> output rejected.
MIN_FIDELITY = 0.85


def run(ctx, sample_ids, min_fidelity=None):
    thr = MIN_FIDELITY if min_fidelity is None else float(min_fidelity)
    models = ctx.pool.get("sidon_enhance", _load, VRAM_GB, ttl_s=300)
    out = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        wav, sr = ctx.load_wav(sid)
        rwav, rsr = _enhance(models, wav, sr)
        corr = round(envelope_corr(wav, sr, rwav, rsr), 3)
        if corr < thr:
            out.append({"src": sid, "rejected": True, "env_corr": corr,
                        "reason": f"restoration diverged from the input take "
                                  f"(envelope corr {corr} < {thr}); output discarded"})
            continue
        meta = {k: v for k, v in ctx.samples[sid].items()
                if k not in ("path", "scores")}
        meta.update({"enhanced_from": sid, "tool": "sidon_enhance",
                     "env_corr": corr})
        nid = ctx.add_sample(rwav, round(len(rwav) / rsr, 2), meta, sr=rsr)
        out.append({"src": sid, "new_sample_id": nid, "env_corr": corr,
                    "dur": round(len(rwav) / rsr, 2), "sr": rsr})
    return {"enhanced": out,
            "note": "restored samples passing the fidelity check are NEW samples; "
                    "re-score them before comparing. rejected=true means Sidon could not "
                    "faithfully restore that take (common on heavily merge-degraded audio)"}
