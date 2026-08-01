"""TTS engine: MOSS-VA-v2 + codec + PEFT multi-adapter LoRA merging + batch generation.

Merging uses the scaling-multiplier pattern proven in the sweep/EVC studies:
each adapter's per-module `scaling` is snapshotted at load, and a merge simply
sets `module.scaling[name] = base * weight` before activating the adapter set.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch


EMOTIONS = [
    "Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe", "Bitterness",
    "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment",
    "Disappointment", "Disgust", "Distress", "Doubt", "Elation", "Embarrassment",
    "Emotional_Numbness", "Fatigue_Exhaustion", "Fear", "Helplessness",
    "Hope_Enthusiasm_Optimism", "Impatience_and_Irritability", "Infatuation", "Interest",
    "Intoxication_Altered_States_of_Consciousness", "Jealousy_and_Envy", "Longing",
    "Malevolence_Malice", "Pain", "Pleasure_Ecstasy", "Pride", "Relief", "Sadness",
    "Sexual_Lust", "Shame", "Sourness", "Teasing", "Thankfulness_Gratitude", "Triumph",
]


def _san(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:48]


class Engine:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.dev = device
        p = cfg["paths"]
        if p["vb_dataset"] not in sys.path:
            sys.path.insert(0, p["vb_dataset"])
        import va_train as vt  # reference processor loader (codec incl.)
        from transformers import AutoModel

        self.pg = vt._gen_processor(cfg["models"]["tts"], cfg["models"]["codec"], device)
        self.sr = int(self.pg.model_config.sampling_rate)
        self.base = AutoModel.from_pretrained(
            cfg["models"]["tts"], trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.n_vq = int(cfg["models"].get("n_vq", 12))

        self.pm = None            # PeftModel once first adapter is loaded
        self._loaded = {}         # adapter_name -> lora catalog name
        self._base_scaling = {}   # adapter_name -> {id(module): base scaling}
        self.active = []          # current merge: [{name, scale}]
        self._char_names = None   # lazy HF listing cache

    # ------------------------------------------------------------- catalog
    def emotion_lora_names(self):
        d = self.cfg["paths"]["emo_loras"]
        return sorted(
            os.path.basename(x) for x in glob.glob(f"{d}/*")
            if os.path.exists(f"{x}/adapter_config.json")
        )

    def vn_lora_names(self):
        d = self.cfg["paths"]["vn_loras"]
        out = []
        for x in sorted(glob.glob(f"{d}/vn_*__*")):
            if os.path.exists(f"{x}/adapter_config.json"):
                dim, side = os.path.basename(x)[3:].rsplit("__", 1)
                out.append(f"vn_{dim}_{side}")
        return out

    def character_lora_names(self):
        """Character LoRA names from the HF repos (cached; offline fallback = local cache)."""
        if self._char_names is not None:
            return self._char_names
        cache = os.path.join(self.cfg["paths"]["char_lora_cache"], "char_names.json")
        names = None
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            files = api.list_repo_files(self.cfg["hf"]["char_repo_genuine"])
            names = sorted({f.split("/")[0] for f in files if "/" in f})
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(names, open(cache, "w"))
        except Exception:
            if os.path.exists(cache):
                names = json.load(open(cache))
            else:  # offline, nothing cached: whatever was downloaded before
                d = self.cfg["paths"]["char_lora_cache"]
                names = sorted(
                    os.path.basename(os.path.dirname(x))
                    for x in glob.glob(f"{d}/*/*/adapter_config.json")
                )
        self._char_names = names
        return names

    def resolve_lora(self, name):
        """Catalog name -> local adapter dir. Families:
        emotion:  '<Emotion>'                    (e.g. 'Fear')
        voicenet: 'vn_<DIM>_<high|low>'          (e.g. 'vn_AROU_high')
        character:'char_genuine/<n>' | 'char_refined/<n>'  (lazy HF download)
        """
        p = self.cfg["paths"]
        if name in EMOTIONS or os.path.isdir(os.path.join(p["emo_loras"], name)):
            d = os.path.join(p["emo_loras"], name)
            if os.path.exists(f"{d}/adapter_config.json"):
                return d
            raise FileNotFoundError(f"emotion LoRA not found: {name}")
        m = re.match(r"^vn_([A-Za-z_]+)_(high|low)$", name)
        if m:
            d = os.path.join(p["vn_loras"], f"vn_{m.group(1)}__{m.group(2)}")
            if os.path.exists(f"{d}/adapter_config.json"):
                return d
            raise FileNotFoundError(f"voicenet LoRA not found: {name}")
        m = re.match(r"^char_(genuine|refined)/(.+)$", name)
        if m:
            repo = self.cfg["hf"][f"char_repo_{m.group(1)}"]
            sub = m.group(2)
            local = os.path.join(p["char_lora_cache"], m.group(1), sub)
            if not os.path.exists(f"{local}/adapter_config.json"):
                from huggingface_hub import hf_hub_download
                os.makedirs(local, exist_ok=True)
                for f in ("adapter_config.json", "adapter_model.safetensors"):
                    src = hf_hub_download(repo, f"{sub}/{f}")
                    tgt = os.path.join(local, f)
                    if not os.path.exists(tgt):
                        os.symlink(src, tgt)
            return local
        raise FileNotFoundError(
            f"unknown LoRA '{name}' (expected an emotion name, vn_<DIM>_<high|low>, "
            f"or char_genuine/<name> | char_refined/<name>)")

    # ------------------------------------------------------------- merging
    def _ensure_loaded(self, name):
        from peft import PeftModel
        an = _san(name)
        if an in self._loaded:
            return an
        path = self.resolve_lora(name)
        if self.pm is None:
            self.pm = PeftModel.from_pretrained(self.base, path, adapter_name=an).eval()
        else:
            self.pm.load_adapter(path, adapter_name=an)
        self._loaded[an] = name
        self._base_scaling[an] = {
            id(m): m.scaling[an] for m in self.pm.modules()
            if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict) and an in m.scaling
        }
        return an

    def merge(self, loras):
        """Activate a merge set: loras = [{name, scale}]. Empty list = plain base model."""
        if not loras:
            if self.pm is not None:
                self.pm.base_model.disable_adapter_layers()
            self.active = []
            return {"active": []}
        names = []
        for l in loras:
            an = self._ensure_loaded(l["name"])
            base = self._base_scaling[an]
            w = float(l["scale"])
            for m in self.pm.modules():
                b = base.get(id(m))
                if b is not None:
                    m.scaling[an] = b * w
            names.append(an)
        self.pm.base_model.enable_adapter_layers()
        self.pm.base_model.set_adapter(names)
        self.active = [{"name": l["name"], "scale": float(l["scale"])} for l in loras]
        return {"active": self.active}

    # ------------------------------------------------------------- reference
    def encode_reference(self, wav_path, max_seconds=15):
        import soundfile as sf
        w, rsr = sf.read(wav_path)
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        w = np.asarray(w, np.float32)[: int(rsr * max_seconds)]
        rc = self.pg.encode_audios_from_wav(
            [torch.as_tensor(w, dtype=torch.float32)], sampling_rate=rsr, n_vq=self.n_vq)[0]
        arr = rc.cpu().numpy() if hasattr(rc, "cpu") else np.asarray(rc)
        if arr.ndim == 2 and arr.shape[0] == self.n_vq and arr.shape[1] != self.n_vq:
            arr = arr.T
        return torch.as_tensor(arr.astype(np.int64)), (w, rsr)

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, text, instruction="", language="English", n=4, ref_codes=None,
                 temp=None, top_p=None, top_k=None, max_frames=None, seed=None):
        """Generate up to n non-empty samples (batched, with empty-retry).
        Returns list of {wav: np.float32 1-D, dur: float}."""
        g = self.cfg["generation"]
        d = g["ref"] if ref_codes is not None else g["no_ref"]
        temp = d["temp"] if temp is None else float(temp)
        top_p = d["top_p"] if top_p is None else float(top_p)
        top_k = d["top_k"] if top_k is None else int(top_k)
        max_frames = int(g["max_frames"] if max_frames is None else max_frames)
        chunk = int(g.get("batch_chunk", 8))
        seed = 1234 if seed is None else int(seed)

        kwargs = dict(text=text, instruction=instruction, language=language)
        if ref_codes is not None:
            kwargs["reference"] = [ref_codes]
        conv = [[self.pg.build_user_message(**kwargs)]]
        bb = self.pg(conv, mode="generation")
        _i = bb["input_ids"].to(self.dev)
        _a = bb["attention_mask"].to(self.dev)
        model = self.pm if (self.pm is not None and self.active) else self.base

        out = []
        attempts = 0
        while len(out) < n and attempts < 3 * n:
            m = min(chunk, n - len(out))
            ii = _i.repeat(m, *([1] * (_i.ndim - 1)))
            am = _a.repeat(m, *([1] * (_a.ndim - 1)))
            torch.manual_seed(seed + 101 * attempts)
            o = model.generate(
                input_ids=ii, attention_mask=am, max_new_frames=max_frames,
                do_sample=True, audio_temperature=temp, audio_top_p=top_p,
                audio_top_k=top_k, audio_repetition_penalty=1.0)
            for msg in self.pg.decode(o):
                if not msg.audio_codes_list:
                    continue
                wav = msg.audio_codes_list[0]
                w = wav.cpu().float().numpy() if hasattr(wav, "cpu") else np.asarray(wav)
                w = w.reshape(-1)
                if w.size < self.sr * 0.4:  # <0.4s = degenerate
                    continue
                out.append({"wav": w.astype(np.float32), "dur": round(w.size / self.sr, 2)})
                if len(out) >= n:
                    break
            attempts += m
        return out

    def gpu_memory_gb(self):
        return round(torch.cuda.memory_allocated() / 1e9, 2)
