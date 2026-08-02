# Model-arm comparison (same missions, same budget)

Judge: Hyprlab gemini-3.6-flash (thinking-native), audio attached: True

| arm | search process | result quality | report clarity | comment |
|---|---|---|---|---|
| 12B | 6 | 5 | 8 | Inefficient tool usage with flatlining or regressing fitness trajectories across runs. Failed to push Explicitness beyond 0.23 (where others hit 0.45+) and suffered high WER penalties on peak arousal/storyteller recipes. |
| 27B | 9 | 10 | 9 | Remarkably search-efficient, completing missions in just 5-7 tool calls. Discovered key structural anchor LoRAs (ARSH, BRGT, NARR) to stabilize speech, achieving peak fitness and near-zero WER across all four dimensions. |
| MoE | 7 | 8 | 6 | Showed strong strategic insight with micro-dose stabilizers (e.g., emotion_Triumph @ 0.05), but failed to complete its Valence mission due to an LLM HTTP 400 crash during report generation, requiring synthetic log recovery. |
| luna | 9 | 9 | 10 | Highly rigorous, systematic search that utilized its budget to conduct deep multi-LoRA dose sweeps. Achieved near-zero WER everywhere and produced the most granular, sample-verified reports of all arms. |

**Winner: 27B** — 27B delivered unmatched search efficiency, needing only 5-7 tool calls per mission to uncover critical stabilizing LoRA combinations (ARSH, BRGT, NARR). It consistently reached peak cohort fitness while maintaining near-zero WER across all four vocal dimensions. Luna was a very close second due to its exhaustive rigor and meticulous reports, while MoE was hindered by an LLM crash.

> 27B: vn_ARSH_high (Arousal Shift) acts as a stabilizer, allowing high-energy delivery while preserving the speech structure.
> 27B: vn_S_NARR_high acts as a stabilizer for vn_S_STRY_high, allowing for higher stylistic expression without the typical collapse in WER associated with pure S_STRY merges.
> MoE: A very small dose of emotion_Triumph (0.05-0.1) helps prevent the 'unraveling' of speech caused by high arousal, effectively acting as a stabilizer for the voice's structure.
> MoE: the agent crashed with an LLM HTTP 400 during report writing after completing all 8 generations
> luna: s0424 — maximum storytelling-style alternative. Recipe: vn_S_STRY_high@0.65 + vn_ARSH_high@0.20 + vn_BRGT_low@0.20. Independent score: S_STRY 0.998, GENU 0.134, BLEND 0.000, ESTH 0.590, RCQL 0.462, QUALITY 0.526, CLRT 0.628, FOCS 0.728, duration 10.96s, WER 0.000.

## Raw verdict JSON
```json
{
 "arms": {
  "12B": {
   "search_process": 6,
   "result_quality": 5,
   "report_clarity": 8,
   "comment": "Inefficient tool usage with flatlining or regressing fitness trajectories across runs. Failed to push Explicitness beyond 0.23 (where others hit 0.45+) and suffered high WER penalties on peak arousal/storyteller recipes."
  },
  "27B": {
   "search_process": 9,
   "result_quality": 10,
   "report_clarity": 9,
   "comment": "Remarkably search-efficient, completing missions in just 5-7 tool calls. Discovered key structural anchor LoRAs (ARSH, BRGT, NARR) to stabilize speech, achieving peak fitness and near-zero WER across all four dimensions."
  },
  "MoE": {
   "search_process": 7,
   "result_quality": 8,
   "report_clarity": 6,
   "comment": "Showed strong strategic insight with micro-dose stabilizers (e.g., emotion_Triumph @ 0.05), but failed to complete its Valence mission due to an LLM HTTP 400 crash during report generation, requiring synthetic log recovery."
  },
  "luna": {
   "search_process": 9,
   "result_quality": 9,
   "report_clarity": 10,
   "comment": "Highly rigorous, systematic search that utilized its budget to conduct deep multi-LoRA dose sweeps. Achieved near-zero WER everywhere and produced the most granular, sample-verified reports of all arms."
  }
 },
 "winner": "27B",
 "verdict": "27B delivered unmatched search efficiency, needing only 5-7 tool calls per mission to uncover critical stabilizing LoRA combinations (ARSH, BRGT, NARR). It consistently reached peak cohort fitness while maintaining near-zero WER across all four vocal dimensions. Luna was a very close second due to its exhaustive rigor and meticulous reports, while MoE was hindered by an LLM crash.",
 "notable_quotes": [
  "27B: vn_ARSH_high (Arousal Shift) acts as a stabilizer, allowing high-energy delivery while preserving the speech structure.",
  "27B: vn_S_NARR_high acts as a stabilizer for vn_S_STRY_high, allowing for higher stylistic expression without the typical collapse in WER associated with pure S_STRY merges.",
  "MoE: A very small dose of emotion_Triumph (0.05-0.1) helps prevent the 'unraveling' of speech caused by high arousal, effectively acting as a stabilizer for the voice's structure.",
  "MoE: the agent crashed with an LLM HTTP 400 during report writing after completing all 8 generations",
  "luna: s0424 \u2014 maximum storytelling-style alternative. Recipe: vn_S_STRY_high@0.65 + vn_ARSH_high@0.20 + vn_BRGT_low@0.20. Independent score: S_STRY 0.998, GENU 0.134, BLEND 0.000, ESTH 0.590, RCQL 0.462, QUALITY 0.526, CLRT 0.628, FOCS 0.728, duration 10.96s, WER 0.000."
 ]
}
```
