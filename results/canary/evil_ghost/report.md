# Mission
Create a convincing 'evil ghost voice': spectral, hollow, menacing, otherworldly, whispery yet threatening. No reference voice. Consider character LoRAs (list them), emotion LoRAs like Malevolence_Malice or Fear, and VoiceNet dims; verify with caption + score, iterate at least twice, save the 2 best samples, then finish reporting the winning recipe.

# Report
### Mission Report: Evil Ghost Voice

**Winning Recipe:**
- **Base Character:** `char_genuine/undead` (Scale: 1.0)
- **Emotions:** `Malevolence_Malice` (Scale: 1.0), `Emotional_Numbness` (Scale: 0.8)
- **VoiceNet Dimensions:** `vn_S_WHIS_high` (Scale: 1.0), `vn_TENS_high` (Scale: 0.8)
- **Prompting Strategy:** 
  - *Description:* "A spectral, hollow, and menacing voice, otherworldly and whispery yet threatening."
  - *Delivery Cue:* "(hissing, echoing with a cold dread)"

**Performance Metrics:**
- **Malevolence_Malice:** 2.148 (Strong menace)
- **S_WHIS (Whisper):** 0.467 (Clear whispery texture)
- **Emotional_Numbness:** 1.191 (Hollow, spectral quality)
- **GENU (Genuineness):** 0.08 (Expected drop for supernatural effects)

**Summary of Iterations:**
1. **Iteration 1 (Undead/Malice/Numbness 0.8):** Best balance of whisper texture and menace. S_WHIS (0.467) was highest here.
2. **Iteration 2 (Demon/Malice/Numbness 0.5):** Provided higher raw Malice but lost the 'ghostly' whisper texture.
3. **Iteration 3 (Undead/Malice/Numbness 0.7 + Nasal):** Nasal resonance added otherworldly color but slightly degraded the whisper clarity.

**Final Recommendation:** Use the Iteration 1 merge. It provides the most consistent 'hollow' and 'menacing' spectral feel without losing the whispery, threatening delivery required for a ghost character.
