# Mission
Produce fear-filled SCREAMING with real vocal bursts. Start from the fear_scream edge recipe genome in your context, push Fear strength and burst blend (BLEND) as high as possible (WER is allowed to be ~1 here), compare at least two variants, save the 2 best samples, then finish with the winning genome.

# Report
Mission complete. Target: Fear-filled screaming with real vocal bursts.

Winning Genome:
- LoRAs: [Fear: 1.5]
- Instruction: GENERAL: A horrified voice, blood-curdling, pushed to the extreme edge of scream.
  SCRIPT:
  (terrified, screaming, voice cracking with panic) <screaming>
- Parameters: temp=1.6, top_p=0.95, top_k=25

Key Findings:
1. The 'Fear' LoRA at 1.5 is the primary driver for the target emotion, reaching strengths >2.4.
2. High temperature (>=1.5) is essential for eliciting non-speech vocal bursts (BLEND) without the LoRA completely collapsing the audio into noise.
3. A hybrid approach (Fear 1.5 + Distress 0.5) was explored but found to be less effective for raw screaming than the pure Fear 1.5 LoRA at high temperature.
4. The best observed BLEND score was 0.448, achieved with a Fear LoRA 1.5 at temp 1.6 using the <screaming> tag.

Final Hall of Fame sample: s0128.
