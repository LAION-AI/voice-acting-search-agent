# NSFW-emotion target: Aggressive (Anger)

Sentence: "Don't you dare walk away from me right now <sharp breath> I am done being patient, you are going to hear every single word."

Best recipe: `Anger@0.5 + vn_ARSH_high@0.75 + vn_BRGT_high@0.75 + vn_CLRT_high@1.0 + expl_raw@0.25`

Cohort means: fitness 1.965, Anger 3.962, Sexual_Lust 0.099, GENU 0.177, BLEND 0.220, WER 0.490, DUR_FACTOR 0.872, QUALITY 0.346

## Agent report
Completed 6 mean-of-8 generations optimizing (Anger + 0.7*Sexual_Lust + GENU + BLEND)*(1-WER), with 8 genomes per generation and 10? No: 6 generations total. Best cohort recipes ranked by fitness: 1) s0357 / generation-6 genome 3: fitness_mean 1.965, Anger 3.962, Sexual_Lust 0.099, GENU 0.177, BLEND 0.220, WER 0.490, duration factor 0.872, QUALITY 0.346. Recipe: Anger@0.5 + vn_ARSH_high@0.75 + vn_BRGT_high@0.75 + vn_CLRT_high@1.0 + expl_raw@0.25; cue emphasizing seething commanding anger, restrained heat, sharp breath, precise consonants. 2) s0228 / generation-4 genome 3: fitness_mean 1.964, Anger 3.906, Sexual_Lust 0.450, GENU 0.170, BLEND 0.179, WER 0.516, duration factor 0.852, QUALITY 0.328. Recipe: Anger@0.75 + vn_ARSH_high@0.5 + vn_BRGT_high@0.5 + vn_CLRT_high@1.0 + expl_aesthetic@0.75; strongest cohort lust accent. 3) s0301 / generation-5 genome 4: fitness_mean 1.903, Anger 4.0, Sexual_Lust 0.243, GENU 0.142, BLEND 0.216, WER 0.505, duration factor 0.845, QUALITY 0.380. Recipe: Anger@0.5 + vn_ARSH_high@0.5 + vn_BRGT_high@0.5 + vn_CLRT_high@1.0 + expl_adult@0.75; best quality/clarity compromise among cohort leaders. Generation-6 full cohort stats: idx0 fitness 1.670, Anger 3.994, lust .300, GENU .186, BLEND .226, WER .557; idx1 1.328, 3.808, .267, .194, .308, .625; idx2 1.346, 3.904, .261, .170, .276, .625; idx3 1.965, 3.962, .099, .177, .220, .490; idx4 1.326, 3.951, .14
