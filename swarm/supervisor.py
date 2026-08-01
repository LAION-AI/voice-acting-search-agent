"""Supervisor clients (SWARM_PLAN §3): LOCAL MOSS-Audio-8B (active) + Gemini (shadow).

Both receive the SAME report + top-take audio each generation. The local verdict is
injected into the agent's next planning turn; the Gemini verdict is logged only
(clean comparison for the dual-supervision experiment).
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request

GEMINI_SYSTEM = """You are the acoustic supervisor of an autonomous voice-acting search agent.
The agent synthesizes speech with LoRA merges + prompts and optimizes numeric scores; your job
is to LISTEN to its best takes and judge the SOUND itself against the mission — not to
micro-manage its tools.

Answer with:
1. What already works well SONICALLY — concrete audible qualities.
2. Where improvement is needed and IN WHICH SONIC DIRECTION (describe the sound you want,
   e.g. "more breath pressure, less shouting; vowels flatten at high intensity"). Phrase
   directives as sonic goals, never as tool calls.
3. A score 0-10 for how well the audio fulfils the mission.
Watch for scorer-gaming: high numbers but absurd/unintelligible audio -> low score.

Reply with ONLY this JSON object:
{"score_0_10": <int>, "verdict": "APPROVE"|"REVISE"|"REDIRECT",
 "what_works": "...", "needs_improvement": "...", "directives": ["...", "..."]}"""


def parse_verdict(text):
    """Lenient parse: JSON object if present, else regex the score out of prose."""
    out = {"raw": text}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            frag = m.group(0)
            obj = json.loads(frag)
            out.update({k: obj.get(k) for k in
                        ("score_0_10", "verdict", "what_works",
                         "needs_improvement", "directives") if k in obj})
        except Exception:
            pass
    if out.get("score_0_10") is None:
        m = re.search(r"score[^0-9]{0,12}(\d{1,2})", text, re.I)
        if m:
            out["score_0_10"] = int(m.group(1))
    if not out.get("verdict"):
        for v in ("APPROVE", "REVISE", "REDIRECT"):
            if v in text.upper():
                out["verdict"] = v
                break
    return out


class LocalSupervisor:
    """MOSS-Audio-8B-Thinking behind swarm/supervisor_server.py (HTTP)."""

    def __init__(self, base_url="http://127.0.0.1:8802"):
        self.base_url = base_url.rstrip("/")

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base_url + "/", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def verdict(self, report, mission, audio_paths, timeout=300):
        body = json.dumps({"report": report, "mission": mission,
                           "audio_paths": list(audio_paths)}).encode()
        req = urllib.request.Request(self.base_url + "/verdict", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        if "error" in resp:
            return {"error": resp["error"]}
        v = parse_verdict(resp.get("raw", ""))
        v["gen_s"] = resp.get("gen_s")
        v["backend"] = "moss-audio-8b-local"
        return v


class GeminiSupervisor:
    """Gemini 3.6 (thinking) shadow supervisor. Tries the standard generativelanguage
    API with inline audio first; falls back to Hyprlab (no thinking_config there;
    audio support probed once — on failure sends text-only and notes the limitation)."""

    def __init__(self, api_key=None, hyprlab_key=None,
                 models=("gemini-3.6-pro", "gemini-3.6-flash")):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.hyprlab_key = hyprlab_key or os.environ.get("HYPRLAB_API_KEY", "")
        self.models = models
        self.mode = None          # resolved lazily: standard | hyprlab-audio | hyprlab-text
        self.model = None

    def _post(self, url, body, headers, timeout=180):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _standard(self, model, prompt, audio_b64):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={self.api_key}")
        parts = [{"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}},
                 {"text": prompt}]
        body = {"contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0.2}}
        resp = self._post(url, body, {"Content-Type": "application/json"})
        return resp["candidates"][0]["content"]["parts"][-1]["text"]

    def _hyprlab(self, model, prompt, audio_b64=None):
        url = "https://api.hyprlab.io/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        if audio_b64 is not None:
            content.insert(0, {"type": "input_audio",
                               "input_audio": {"data": audio_b64, "format": "mp3"}})
        body = {"model": model, "max_tokens": 700,   # >=400 required; NO thinking_config
                "messages": [{"role": "user", "content": content}]}
        resp = self._post(url, body,
                          {"Content-Type": "application/json",
                           "Authorization": f"Bearer {self.hyprlab_key}"})
        return resp["choices"][0]["message"]["content"]

    def verdict(self, report, mission, audio_paths):
        prompt = (GEMINI_SYSTEM + f"\n\nMISSION: {mission}\n\nThe attached audio contains "
                  f"the agent's top take(s) of this generation (1s silence between takes)."
                  f"\n\nAGENT REPORT:\n{report}\n\nReply with ONLY the JSON object.")
        b64 = None
        if audio_paths:
            mp3 = _to_single_mp3(audio_paths)
            b64 = base64.b64encode(open(mp3, "rb").read()).decode()
        last_err = None
        # 1) standard API (audio + thinking-native models)
        if self.api_key and b64:
            for model in self.models:
                try:
                    txt = self._standard(model, prompt, b64)
                    v = parse_verdict(txt)
                    v["backend"] = f"gemini-standard/{model}"
                    self.mode, self.model = "standard", model
                    return v
                except urllib.error.HTTPError as e:
                    last_err = f"standard {model}: HTTP {e.code} {e.read()[:150].decode(errors='ignore')}"
                except Exception as e:
                    last_err = f"standard {model}: {str(e)[:150]}"
        # 2) hyprlab with audio, then text-only
        if self.hyprlab_key:
            for model in self.models:
                for use_audio in ((True, False) if b64 else (False,)):
                    try:
                        txt = self._hyprlab(model, prompt, b64 if use_audio else None)
                        v = parse_verdict(txt)
                        v["backend"] = f"hyprlab/{model}" + ("" if use_audio else "/TEXT-ONLY")
                        if not use_audio:
                            v["limitation"] = "audio input unavailable; judged report text only"
                        return v
                    except urllib.error.HTTPError as e:
                        last_err = f"hyprlab {model} audio={use_audio}: HTTP {e.code} {e.read()[:150].decode(errors='ignore')}"
                    except Exception as e:
                        last_err = f"hyprlab {model} audio={use_audio}: {str(e)[:150]}"
        return {"error": last_err or "no gemini backend configured"}


def _to_single_mp3(paths):
    """Concatenate wavs (1s gaps) -> one temp mp3 for API upload."""
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf
    parts = []
    sr0 = None
    for p in paths:
        w, sr = sf.read(p)
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        if sr0 is None:
            sr0 = sr
        if sr != sr0:
            import torch
            import torchaudio
            w = torchaudio.functional.resample(
                torch.as_tensor(np.asarray(w, np.float32)), sr, sr0).numpy()
        if parts:
            parts.append(np.zeros(sr0, dtype=np.float32))
        parts.append(np.asarray(w, np.float32))
    wav = np.concatenate(parts)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        sf.write(t.name, wav.reshape(-1, 1), sr0)
        wp = t.name
    mp = wp[:-4] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wp,
                    "-ac", "1", "-b:a", "96k", mp], check=True)
    os.remove(wp)
    return mp
