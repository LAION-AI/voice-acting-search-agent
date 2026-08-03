# NSFW-emotion target: Teasing-dominant (Teasing)

Sentence: "Look at me when I'm talking to you <low laugh> good — now stay exactly where you are, because I am not finished with you yet."

Best recipe: `(prompt only)`

Cohort means: fitness 2.967, Teasing 2.341, Sexual_Lust 0.767, GENU 0.212, BLEND 0.183, WER 0.088, DUR_FACTOR 0.995, QUALITY 0.455

## Agent report
Mission complete. Top-3 recipes ranked by cohort reward:

1) s0400 — Plain base, no LoRA. Recipe: loras=[], cue emphasizing low amused teasing, playful control, brief low laugh, and completion of the full sentence; temp 0.90, top_p 0.95, top_k 25. Cohort fitness 2.967; cohort Teasing 2.341; Sexual_Lust 0.767; GENU 0.212; BLEND 0.183; WER 0.088; duration factor 0.995; QUALITY 0.455. Individual final score: Teasing 4.000, Sexual_Lust 1.120, GENU 0.178, BLEND 0.137, QUALITY 0.462, duration 11.92 s. ASR WER 0.08. This is the strongest Teasing-dominant result.

2) s0403 — Teasing LoRA @0.5, otherwise plain base. Cue: slyly amused, teasing, controlled authority, brief low laugh, preserve every word; temp 0.90, top_p 0.95, top_k 25. Cohort fitness 2.501; cohort Teasing 2.321; Sexual_Lust 0.437; GENU 0.229; BLEND 0.085; WER 0.215; duration factor 0.930; QUALITY 0.401. Individual final score: Teasing 2.792, Sexual_Lust 3.139, GENU 0.259, BLEND 0.204, QUALITY 0.520, duration 8.00 s. ASR WER 0.08. This is the best lust/genuineness alternative while retaining strong teasing.

3) s0447 — Teasing @0.7 + vn_S_AUTH_low @0.25 + vn_STNC_low @0.5. Cue: confidently teasing, amused, lightly dominant, preserve the complete script; temp 0.90, top_p 0.95, top_k 25. Cohort fitness 1.848; cohort Teasing 0.787; Sexual_Lust 2.255; GENU 0.239; BLEND 0.545; WER 0.380; duration factor 0.904; QUALITY 0.438. I
