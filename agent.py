"""Self-coded ReAct JSON tool loop: robust parsing with retries, budget enforcement,
context-overflow compression, persistent memory, JSONL transcripts, subagent spawning.

The LLM must answer every turn with ONE JSON object:
    {"thought": "...", "tool": "<tool name>", "args": {...}}
Structured output (vLLM json_schema) guarantees parseability; a bracket-matching
fallback parser plus feedback-retries covers plain-text mode too.
"""
import json
import os
import time

import tools as T


TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


def extract_json(text):
    """Parse the first balanced JSON object out of an LLM reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in reply")


class Agent:
    def __init__(self, cfg, llm, ctx, get_system_context, mission, budget,
                 transcript_path, name="main", allow_spawn=True):
        self.cfg = cfg
        self.llm = llm
        self.ctx = ctx
        self.get_system_context = get_system_context
        self.mission = mission
        self.budget = int(budget)
        self.name = name
        self.transcript_path = transcript_path
        self.allow_spawn = allow_spawn
        acfg = cfg["agent"]
        self.max_json_retries = int(acfg["max_json_retries"])
        self.max_context_tokens = int(acfg["max_context_tokens"])
        self.result_cap = int(acfg["tool_result_char_cap"])
        self.messages = []  # excluding system

    # ------------------------------------------------------------ plumbing
    def log(self, kind, payload):
        rec = {"t": round(time.time(), 2), "agent": self.name, "kind": kind}
        rec.update(payload if isinstance(payload, dict) else {"data": payload})
        os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _system(self):
        sc = self.get_system_context()
        return (
            f"{sc}\n\n"
            "## Response protocol\n"
            "Reply with EXACTLY ONE JSON object per turn, no prose outside it:\n"
            '{"thought": "<brief reasoning>", "tool": "<tool_name>", "args": {...}}\n'
            "When the mission is complete (or the budget forces it), call the `finish` "
            'tool: {"thought": "...", "tool": "finish", "args": {"report": "<final report>"}}.\n'
            f"You have a budget of tool calls; {self.budget} remain at mission start. "
            "Spend them deliberately: batch work (generate n>1, score lists, run_generation) "
            "and record key findings with the memory tool."
        )

    def _estimate_tokens(self, msgs):
        # ~2.7-3 chars/token for markdown-heavy English; use 3 to stay conservative
        return sum(len(m["content"]) for m in msgs) // 3

    def _compress(self):
        """Keep mission + recent turns; splice older turns out with a note + memory tail."""
        if len(self.messages) <= 10:
            return
        head = self.messages[:1]            # the mission message
        tail = self.messages[-8:]
        mem = ""
        if os.path.exists(self.ctx.memory_path):
            mem = open(self.ctx.memory_path).read()[-3000:]
        note = ("[context compressed: older turns removed. Persistent memory follows]\n"
                + mem)
        self.messages = head + [{"role": "user", "content": note}] + tail
        self.log("compress", {"kept": len(self.messages)})

    def _chat(self):
        msgs = [{"role": "system", "content": self._system()}] + self.messages
        if self._estimate_tokens(msgs) > self.max_context_tokens:
            self._compress()
            msgs = [{"role": "system", "content": self._system()}] + self.messages
        try:
            return self.llm.chat(msgs, json_schema=TOOL_CALL_SCHEMA)
        except Exception:
            # structured-output path failed -> plain completion, parser handles it
            return self.llm.chat(msgs)

    # ------------------------------------------------------------ main loop
    def run(self):
        if self.allow_spawn:
            self.ctx.spawn_fn = self._spawn
        self.messages.append({"role": "user", "content": f"MISSION: {self.mission}"})
        self.log("mission", {"mission": self.mission, "budget": self.budget})
        used = 0
        report = None
        parse_fails = 0
        while True:
            reply = self._chat()
            self.log("llm", {"reply": reply[:4000]})
            try:
                call = extract_json(reply)
                tool = call.get("tool")
                args = call.get("args", {})
                if tool not in T.TOOLS:
                    raise ValueError(f"unknown tool '{tool}'; available: {list(T.TOOLS)}")
                parse_fails = 0
            except Exception as ex:
                parse_fails += 1
                self.log("parse_error", {"error": str(ex)})
                if parse_fails > self.max_json_retries:
                    report = f"[aborted: {parse_fails} consecutive malformed replies]"
                    break
                self.messages.append({"role": "assistant", "content": reply[:2000]})
                self.messages.append({"role": "user", "content":
                    f"FORMAT ERROR: {ex}. Reply again with exactly one JSON object "
                    '{"thought", "tool", "args"}.'})
                continue

            self.messages.append({"role": "assistant", "content": json.dumps(call)})
            if tool == "finish":
                hof = os.path.join(self.ctx.workdir, "hall_of_fame")
                if (self.ctx.samples and not getattr(self, "_hof_nudged", False)
                        and os.path.isdir(hof) and not os.listdir(hof)):
                    self._hof_nudged = True
                    self.messages.append({"role": "user", "content":
                        "Before finishing: persist your best samples with save_best "
                        "(the hall of fame is still empty), then call finish again."})
                    continue
                report = str(args.get("report", ""))
                self.log("finish", {"report": report})
                break

            if used >= self.budget:
                self.messages.append({"role": "user", "content":
                    "BUDGET EXHAUSTED. You must now call the finish tool with your "
                    "final report summarizing findings, best genomes and hall-of-fame samples."})
                # allow a couple of attempts to produce finish, else abort
                if used >= self.budget + 2:
                    report = "[aborted: budget exhausted and no finish produced]"
                    break
                used += 1
                continue

            if tool == "spawn_subagent" and not self.allow_spawn:
                result = {"error": "subagents cannot spawn further subagents"}
            else:
                t0 = time.time()
                result = T.run_tool(self.ctx, tool, args)
                result_s = json.dumps(result, default=str)
                self.log("tool", {"tool": tool, "args": args,
                                  "seconds": round(time.time() - t0, 1),
                                  "result": json.loads(result_s) if len(result_s) < 20000
                                  else {"truncated": result_s[:20000]}})
            used += 1
            result_s = json.dumps(result, default=str)
            if len(result_s) > self.result_cap:
                result_s = result_s[:self.result_cap] + f'... [truncated {len(result_s)} chars]"}}'
            self.messages.append({"role": "user", "content":
                f"TOOL_RESULT {tool} (calls used {used}/{self.budget}):\n{result_s}"})

        self.log("done", {"tool_calls_used": used})
        return {"report": report, "tool_calls_used": used,
                "transcript": self.transcript_path}

    # ------------------------------------------------------------ subagents
    def _spawn(self, task, budget):
        sub_name = f"{self.name}.sub{int(time.time()) % 100000}"
        tpath = os.path.join(os.path.dirname(self.transcript_path),
                             f"transcript_{sub_name}.jsonl")
        sub = Agent(self.cfg, self.llm, self.ctx, self.get_system_context,
                    mission=task, budget=budget, transcript_path=tpath,
                    name=sub_name, allow_spawn=False)
        res = sub.run()
        return {"subagent": sub_name, "report": res["report"],
                "tool_calls_used": res["tool_calls_used"]}
