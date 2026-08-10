# Adaptive Hybrid Attention — Reproducible Synthetic Test

Command:
python adaptive_hybrid_attention.py --trials 200 --noise 1.25

Observed output:

| Metric | Location-sensitive | Adaptive Hybrid |
|---|---:|---:|
| Backward-step rate | 0.3265 | 0.2934 |
| Large-jump rate | 0.0796 | 0.0516 |
| Mean absolute alignment error | 1.1528 | 1.0235 |
| Attention entropy | 1.4472 | 1.6199 |

Relative changes:
- Backward-step rate: about 10.1% lower
- Large-jump rate: about 35.2% lower
- Mean absolute alignment error: about 11.2% lower
- Attention entropy: about 11.9% higher

Interpretation:
The controlled test supports the hypothesis that combining a monotonic prior with
location-sensitive attention can reduce backward motion and large alignment jumps.
This is NOT an LJSpeech/VCTK TTS quality result and must not be presented as MOS,
CER, or real speech-synthesis evidence.

For a paper submission, the next stage should train a complete Tacotron2-style
model on a real corpus and compare identical baselines using MOS, CER/WER,
alignment-failure rate, long-sentence robustness, MCD (if desired), and inference time.
