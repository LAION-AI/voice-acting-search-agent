"""Scorer stack: 99-vec (EmoNet-40 + VoiceNet-57 + genuineness + burst-blend),
WER via ASR (Parakeet if available, else faster-whisper x3 decode variants),
procedural captions, and ECAPA speaker similarity.

Reuses the reference implementations verbatim:
  - va_rescore.NinetyNineScorer (paths.vb_dataset) for the 99-vec models
  - spk_sim.SpkSim (paths.vb_dataset) for ECAPA
  - procedural-voice-captions caption.py (paths.pvc_repo) for captions
"""
import json
import re
import sys

import numpy as np
import torch


# emotion-name mapping: scorer names (underscored) -> caption-baseline names
_EMO_NAME_OVERRIDES = {
    "Hope_Enthusiasm_Optimism": "Hope/Optimism",
    "Intoxication_Altered_States_of_Consciousness": "Intoxication/Altered States",
}


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def wer(ref, hyp):
    """Word error rate via Levenshtein distance on normalized words."""
    def toks(s):
        return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
    r, h = toks(ref), toks(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(min(1.0, d[len(r), len(h)] / len(r)))


class ScorerStack:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.dev = device
        p = cfg["paths"]
        for path in (p["vb_dataset"], p["pvc_repo"]):
            if path not in sys.path:
                sys.path.insert(0, path)
        import va_rescore
        self.nns = va_rescore.NinetyNineScorer(device=device)
        self.layout = json.load(open(p["score_layout"]))
        self.slot_code = [s.get("code") or s["block"].upper() for s in self.layout["slots"]]
        # GENU / BLEND aliases for slots 97/98
        self.slot_code[97] = "GENU"
        self.slot_code[98] = "BLEND"

        import caption as pvc
        self._pvc = pvc
        self._pvc_baseline = pvc.load_baseline()
        # map scorer emotion names -> caption baseline emotion names
        base_emos = {k for k, v in self._pvc_baseline.items()
                     if isinstance(v, dict) and v.get("group") == "emonet"}
        by_norm = {_norm_key(k): k for k in base_emos}
        self.emo_name_map = {}
        for e in list(self.nns.emo_experts.keys()):
            tgt = _EMO_NAME_OVERRIDES.get(e) or by_norm.get(_norm_key(e))
            if tgt:
                self.emo_name_map[e] = tgt

        self._spk = None
        self._asr = None
        self._asr_backend = None

        # EIV-Plus content-enjoyment head (same FullEmbeddingMLP family as the 40
        # emotion experts, laion/Empathic-Insight-Voice-Plus) -> score code ENJOY.
        try:
            from collections import OrderedDict
            from huggingface_hub import hf_hub_download
            import score_emotions as se_mod
            pth = hf_hub_download("laion/Empathic-Insight-Voice-Plus",
                                  "model_score_content_enjoyment_best.pth")
            m = se_mod.FullEmbeddingMLP().to(device)
            sd_ = torch.load(pth, map_location=device)
            if any(k.startswith("_orig_mod.") for k in sd_):
                sd_ = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd_.items())
            m.load_state_dict(sd_)
            m.eval()
            m.half()
            self.nns.emo_experts["score_content_enjoyment"] = m
            self.has_enjoy = True
            print("[scorers] ENJOY head loaded (EIV-Plus content enjoyment)")
        except Exception as ex:
            self.has_enjoy = False
            print(f"[scorers] ENJOY head unavailable: {str(ex)[:120]}")

    # ---------------------------------------------------------------- 99-vec
    @torch.no_grad()
    def score_full(self, wav, sr):
        """Like NinetyNineScorer.score_wav but ALSO returns the raw per-model values
        (needed for procedural captions, which expect raw 0-6 scales)."""
        sc = self.nns
        w16 = sc._to_mono_np(wav, sr, 16000)
        wt16 = torch.from_numpy(w16)
        emo_raw = sc.se.score_waveforms([w16], sc.emo_proc, sc.emo_enc,
                                        sc.emo_experts, sc.dev)[0]
        emb_vn = sc.vn.embed((wt16.unsqueeze(0), 16000))
        vn_raw = {}
        for code in sc.vn_codes:
            r = sc.vn.reg[code]
            xr = ((emb_vn - r["mu"]) / r["sd"]).unsqueeze(0)
            vn_raw[code] = float(r["net"](xr).squeeze().cpu())
        e = sc.vc.encode_waveform(wt16.to(sc.dev))
        e = e[0] if e.dim() == 2 else e
        e = e.float().unsqueeze(0).cpu()
        gm, gmu, gsd = sc.genu
        bm, bmu, bsd = sc.blend
        genu_raw = float(gm((e - gmu) / gsd))
        blend_raw = float(bm((e - bmu) / bsd))

        vec = np.zeros(99, dtype=np.float32)
        for i, (block, code, norm) in enumerate(sc.plan):
            if block == "emotion":
                val = emo_raw[code]
            elif block == "voicenet":
                val = vn_raw[code]
            elif block == "genuineness":
                val = genu_raw
            else:
                val = blend_raw
            vec[i] = sc._norm_slot(val, norm)
        return {"vec": vec, "emo_raw": emo_raw, "vn_raw": vn_raw,
                "genu_raw": genu_raw, "blend_raw": blend_raw,
                "enjoy_raw": emo_raw.get("score_content_enjoyment")}

    def slot_value(self, vec, code):
        """Normalized value of a slot by code ('AROU', 'Fear', 'GENU', 'BLEND', ...)."""
        try:
            return float(vec[self.slot_code.index(code)])
        except ValueError:
            raise KeyError(f"unknown score code '{code}'")

    def quality(self, vec):
        """Speech-quality proxy: mean of RCQL + ESTH normalized slots."""
        return float((self.slot_value(vec, "RCQL") + self.slot_value(vec, "ESTH")) / 2)

    # ---------------------------------------------------------------- caption
    def caption(self, full_scores, seed=0):
        preds = {
            "dims": dict(full_scores["vn_raw"]),
            "emo": {self.emo_name_map[k]: v for k, v in full_scores["emo_raw"].items()
                    if k in self.emo_name_map},
            "genu": full_scores["genu_raw"],
            "blend": full_scores["blend_raw"],
        }
        return self._pvc.caption(preds, self._pvc_baseline, synonym_seed=seed)

    # ---------------------------------------------------------------- ASR/WER
    def _ensure_asr(self):
        if self._asr is not None:
            return
        backend = self.cfg["asr"].get("backend", "auto")
        if backend in ("auto", "parakeet"):
            try:  # graceful degradation: Parakeet needs nemo, usually unavailable
                import nemo.collections.asr as nemo_asr  # noqa
                self._asr = nemo_asr.models.ASRModel.from_pretrained(
                    "nvidia/parakeet-tdt-0.6b-v3")
                self._asr_backend = "parakeet"
                return
            except Exception:
                if backend == "parakeet":
                    raise
        from faster_whisper import WhisperModel
        self._asr = WhisperModel(self.cfg["asr"].get("whisper_model", "small"),
                                 device="cuda", compute_type="float16")
        self._asr_backend = "faster-whisper"

    def transcribe(self, wav, sr, ref_text=None, language="en"):
        """3 decode variants -> transcripts + per-variant WER vs ref_text (if given).
        Returns {backend, transcripts, wers, wer_mean, wer_median}."""
        self._ensure_asr()
        w16 = self.nns._to_mono_np(wav, sr, 16000)
        transcripts = []
        if self._asr_backend == "parakeet":
            out = self._asr.transcribe([w16])
            transcripts = [out[0].text if hasattr(out[0], "text") else str(out[0])]
        else:
            for var in self.cfg["asr"].get("variants", [{"beam_size": 5}]):
                kw = dict(language=language, beam_size=int(var.get("beam_size", 1)))
                if "temperature" in var:
                    kw["temperature"] = float(var["temperature"])
                segs, _ = self._asr.transcribe(w16, **kw)
                transcripts.append(" ".join(s.text.strip() for s in segs).strip())
        res = {"backend": self._asr_backend, "transcripts": transcripts}
        if ref_text is not None:
            wers = [round(wer(ref_text, t), 4) for t in transcripts]
            res.update(wers=wers, wer_mean=round(float(np.mean(wers)), 4),
                       wer_median=round(float(np.median(wers)), 4))
        return res

    # ---------------------------------------------------------------- spk sim
    def _ensure_spk(self):
        if self._spk is None:
            import spk_sim
            self._spk = spk_sim.SpkSim(device=self.dev)
        return self._spk

    def speaker_sim(self, wav, sr, ref_wav, ref_sr):
        return float(self._ensure_spk().cos(wav, sr, ref_wav, ref_sr))
