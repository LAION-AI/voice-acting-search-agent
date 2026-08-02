#!/usr/bin/env python3
"""Local audio-LM supervisor server (SWARM_PLAN §3.3 `local-audio-lm` backend).

Loads MOSS-Audio-8B-Thinking (no-think mode: ~same quality, ~5x faster) and serves
POST /verdict  {"report": str, "mission": str, "audio_paths": [wav|mp3, ...]}
            -> {"raw": str, ...parsed verdict fields}

MUST run in the `audio_processing` conda env (transformers 4.57.x — MOSS audio code
breaks on transformers 5.x) with a clean environment (`env -i`), e.g.:

  env -i HOME=$HOME PATH=/home/deployer/miniconda3/envs/audio_processing/bin:/usr/bin:/bin \
      CUDA_VISIBLE_DEVICES=2 HF_HOME=/tmp/hf_cache \
      python swarm/supervisor_server.py --port 8802
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/run/user/1001/moss_think/MOSS-Audio")
sys.path.insert(0, "/run/user/1001/moss_think")

import numpy as np
import torch

MODEL_PATH = "/run/user/1001/moss_think/weights/MOSS-Audio-8B-Thinking"

SYSTEM = """You are an acoustic QUALITY RATER for an autonomous voice-acting search agent.
You LISTEN to its best takes and judge how well the SOUND fulfils the mission.

Strict rules for your answer:
- Report ONLY what you actually hear in THIS audio. Every observation must reference a
  concrete audible moment or property of these specific takes (quote the words you hear or
  describe the exact sound event). Do NOT give advice, directions, or suggestions of any kind.
- If the speech is unintelligible, slurred, or obviously artificial, say so plainly and
  score low - high classifier numbers in the report do not override your ears.
- Score 0-10 for mission fit (10 = production-ready).

Reply with ONLY this JSON object:
{"score_0_10": <int>,
 "observations": ["<concrete audible fact 1>", "<concrete audible fact 2>", ...]}"""


def load_model():
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor
    t0 = time.time()
    model = MossAudioModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
    proc = MossAudioProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, enable_time_marker=True)
    print(f"[supervisor] MOSS-Audio-8B loaded in {time.time()-t0:.0f}s", flush=True)
    return model, proc


def concat_audio(paths, sr):
    """Concatenate takes with 1s silence gaps -> one waveform at `sr`."""
    import soundfile as sf
    import torchaudio
    parts = []
    gap = np.zeros(sr, dtype=np.float32)
    for p in paths:
        w, s = sf.read(p)
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        w = np.asarray(w, np.float32)
        if s != sr:
            w = torchaudio.functional.resample(torch.as_tensor(w), s, sr).numpy()
        if parts:
            parts.append(gap)
        parts.append(w)
    return np.concatenate(parts) if parts else gap


def build_prompt(report, mission, n_takes):
    user = (
        f"MISSION: {mission}\n\n"
        f"The attached audio contains the agent's TOP {n_takes} take(s) of this "
        f"generation, separated by 1 second of silence.\n\n"
        f"AGENT REPORT (its own numbers and interpretation):\n{report}\n\n"
        "Listen carefully to the takes, then give your acoustic assessment as the "
        "single JSON object specified in your instructions.")
    return (
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")


class Handler(BaseHTTPRequestHandler):
    model = None
    proc = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            sr = int(self.proc.config.mel_sr)
            aud = concat_audio(req.get("audio_paths", []), sr)
            prompt = build_prompt(req.get("report", ""), req.get("mission", ""),
                                  len(req.get("audio_paths", [])))
            inp = self.proc(text=prompt, audios=[aud], return_tensors="pt").to(self.model.device)
            inp["audio_data"] = inp["audio_data"].to(self.model.dtype)
            inp["audio_input_mask"] = inp["input_ids"] == self.proc.audio_token_id
            t0 = time.time()
            with torch.no_grad():
                g = self.model.generate(**inp, max_new_tokens=600, do_sample=False,
                                        use_cache=True)
            out = self.proc.decode(g[0, inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            resp = {"raw": out, "gen_s": round(time.time() - t0, 1)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        except Exception as ex:
            import traceback
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps(
                {"error": str(ex)[:300], "trace": traceback.format_exc()[-500:]}).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8809)
    args = ap.parse_args()
    Handler.model, Handler.proc = load_model()
    print(f"[supervisor] serving on :{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
