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
        self.temperature = float(cfg["llm"].get("temperature", 0.7))
        self.max_tokens = int(cfg["llm"].get("max_tokens", 2048))

    def _post(self, path, payload, timeout=600):
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def healthy(self):
        try:
            req = urllib.request.Request(self.base_url + "/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                json.loads(r.read())
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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "tool_call", "schema": json_schema},
            }
        last = None
        for attempt in range(3):
            try:
                out = self._post("/chat/completions", payload)
                return out["choices"][0]["message"]["content"]
            except Exception as e:  # transient server errors -> retry
                last = e
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
        "--disable-log-requests",
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
