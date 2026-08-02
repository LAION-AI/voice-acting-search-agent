"""vLLM OpenAI-compatible client + server management.

The LLM brain (Gemma-4-12B-it QAT w4a16) is served by vLLM with a LOW
gpu_memory_utilization so the TTS model + scorer stack fit next to it on the
same 80GB GPU (see configs/single_gpu.yaml).  This module only needs `requests`.
"""
import json
import os
import subprocess
import time
import urllib.request


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal OpenAI-compatible /chat/completions client (no `openai` dep)."""

    def __init__(self, cfg):
        self.base_url = cfg["llm"]["base_url"].rstrip("/")
        self.model = cfg["llm"]["model"]
        self.api_key = os.environ.get("LLM_API_KEY", "")
        # External OpenAI-compatible APIs (Hyprlab etc.): LLM_MODEL pins the model —
        # do NOT auto-detect from /models (such providers list hundreds of models).
        env_model = os.environ.get("LLM_MODEL")
        if env_model:
            self.model = env_model
            print(f"[llm] model pinned via LLM_MODEL: {self.model}")
        else:
            try:  # local multi-arm: trust whatever model the server actually serves
                req = urllib.request.Request(self.base_url + "/models")
                with urllib.request.urlopen(req, timeout=5) as r:
                    served = [m["id"] for m in json.loads(r.read()).get("data", [])]
                if served and self.model not in served:
                    print(f"[llm] serving {served[0]} (config said {self.model})")
                    self.model = served[0]
            except Exception:
                pass
        # reasoning models: bill thinking as OUTPUT tokens -> default "low" keeps cost
        # down; tool-call JSON planning does not need deep reasoning. Override with
        # LLM_REASONING_EFFORT (low|medium|high|none). Only sent when set.
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT") or None
        self.temperature = float(cfg["llm"].get("temperature", 0.7))
        self.max_tokens = int(cfg["llm"].get("max_tokens", 2048))
        self.last_usage = None
        self.usage_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _post(self, path, payload, timeout=600):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def healthy(self):
        # External OpenAI-compatible APIs (Hyprlab etc.) require the auth header even on
        # /models; a keyed request also covers vLLM (which just ignores the header).
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(self.base_url + "/models", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                json.loads(r.read())
            return True
        except Exception:
            # Fall back to a 1-token chat probe: some gateways gate /models but serve
            # /chat/completions, or pin a model not in the listing.
            try:
                self.chat([{"role": "user", "content": "ping"}], max_tokens=1)
                return True
            except Exception:
                return False

    def chat(self, messages, temperature=None, max_tokens=None, json_schema=None):
        """messages: [{role, content}] -> assistant text.  If `json_schema` is given,
        vLLM structured output guarantees the reply parses against it."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_schema is not None:
            # standard OpenAI response_format shape — works for vLLM AND external
            # OpenAI-compatible APIs with structured-output support
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "tool_call", "schema": json_schema},
            }
        if self.reasoning_effort and self.reasoning_effort != "none":
            payload["reasoning_effort"] = self.reasoning_effort
        last = None
        for attempt in range(3):
            try:
                out = self._post("/chat/completions", payload)
                u = out.get("usage")
                if u:
                    self.last_usage = u
                    self.usage_totals["calls"] += 1
                    self.usage_totals["prompt_tokens"] += int(u.get("prompt_tokens", 0))
                    self.usage_totals["completion_tokens"] += int(u.get("completion_tokens", 0))
                return out["choices"][0]["message"]["content"]
            except Exception as e:  # transient server errors -> retry
                last = e
                if "reasoning_effort" in payload and attempt == 0:
                    payload.pop("reasoning_effort")  # provider may reject the param
                time.sleep(2 * (attempt + 1))
        raise LLMError(f"LLM chat failed after retries: {last}")


def start_server(cfg, log_path, env_extra=None):
    """Spawn the vLLM OpenAI server as a subprocess on the SAME visible GPU.
    Returns the Popen handle. Caller should poll LLMClient.healthy()."""
    llm = cfg["llm"]
    cmd = [
        llm["vllm_python"], "-m", "vllm.entrypoints.openai.api_server",
        "--model", llm["model"],
        "--port", str(llm["port"]),
        "--gpu-memory-utilization", str(llm["gpu_memory_utilization"]),
        "--max-model-len", str(llm["max_model_len"]),
    ]
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    logf = open(log_path, "ab")
    return subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)


def wait_healthy(client, proc=None, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if client.healthy():
            return True
        if proc is not None and proc.poll() is not None:
            raise LLMError(f"vLLM server exited with code {proc.returncode}")
        time.sleep(5)
    return False
