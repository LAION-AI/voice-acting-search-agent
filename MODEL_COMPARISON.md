# Model-arm comparison (same missions, same budget)

Judge: Hyprlab gemini-3.6-flash (thinking-native), audio attached: True

| arm | search process | result quality | report clarity | comment |
|---|---|---|---|---|
| 12B | 6 | 5 | 8 | 12B consumed 42 tool calls yet struggled to discover high-performing recipes, landing on vastly lower target dimensions (VALN 0.555, EXPL 0.232) than peers. Trajectories often flatlined or degraded over generations (e.g. Storyteller), though its reports were tidy and well-structured. |
| 27B | 10 | 10 | 9 | 27B demonstrated masterclass search efficiency, completing all 4 missions in just 22 tool calls while finding high-synergy multi-adapter combinations. It consistently reached peak dimension scores with near-zero WER penalties by systematically discovering stabilization adapters like vn_ARSH_high and vn_BRGT_high. |
| MoE | 7 | 8 | 7 | MoE achieved high raw target scores (e.g. VALN 0.808), but its search process was noisy with frequent fitness regressions across 43+ tool calls. Crucially, it crashed with an LLM HTTP 400 error during the Valence mission report phase, requiring fallback log synthesis. |

**Winner: 27B** — 27B overwhelmingly won the evaluation by delivering maximum search efficiency and superior audio quality in just 22 total tool calls. It systematically uncovered critical multi-LoRA stabilization rules (e.g., using Arousal Shift and Brightness adapters to protect intelligibility), while 12B underperformed on target metrics and MoE suffered from noisy search dynamics and an unhandled API crash.

> 27B: vn_ARSH_high (Arousal Shift) acts as a stabilizer, allowing high-energy delivery while preserving the speech structure.
> 27B: The synergy between VALN and BRGT VoiceNet adapters is the most reliable path to high-valence, organic speech.
> MoE: (the agent crashed with an LLM HTTP 400 during report writing after completing all 8 generations; this report was generated programmatically from evolution_log.jsonl

## Raw verdict JSON
```json
{
 "arms": {
  "12B": {
   "search_process": 6,
   "result_quality": 5,
   "report_clarity": 8,
   "comment": "12B consumed 42 tool calls yet struggled to discover high-performing recipes, landing on vastly lower target dimensions (VALN 0.555, EXPL 0.232) than peers. Trajectories often flatlined or degraded over generations (e.g. Storyteller), though its reports were tidy and well-structured."
  },
  "27B": {
   "search_process": 10,
   "result_quality": 10,
   "report_clarity": 9,
   "comment": "27B demonstrated masterclass search efficiency, completing all 4 missions in just 22 tool calls while finding high-synergy multi-adapter combinations. It consistently reached peak dimension scores with near-zero WER penalties by systematically discovering stabilization adapters like vn_ARSH_high and vn_BRGT_high."
  },
  "MoE": {
   "search_process": 7,
   "result_quality": 8,
   "report_clarity": 7,
   "comment": "MoE achieved high raw target scores (e.g. VALN 0.808), but its search process was noisy with frequent fitness regressions across 43+ tool calls. Crucially, it crashed with an LLM HTTP 400 error during the Valence mission report phase, requiring fallback log synthesis."
  }
 },
 "winner": "27B",
 "verdict": "27B overwhelmingly won the evaluation by delivering maximum search efficiency and superior audio quality in just 22 total tool calls. It systematically uncovered critical multi-LoRA stabilization rules (e.g., using Arousal Shift and Brightness adapters to protect intelligibility), while 12B underperformed on target metrics and MoE suffered from noisy search dynamics and an unhandled API crash.",
 "notable_quotes": [
  "27B: vn_ARSH_high (Arousal Shift) acts as a stabilizer, allowing high-energy delivery while preserving the speech structure.",
  "27B: The synergy between VALN and BRGT VoiceNet adapters is the most reliable path to high-valence, organic speech.",
  "MoE: (the agent crashed with an LLM HTTP 400 during report writing after completing all 8 generations; this report was generated programmatically from evolution_log.jsonl"
 ]
}
```
