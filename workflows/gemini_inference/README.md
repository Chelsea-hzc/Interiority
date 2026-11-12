# Gemini corpus inference

This workflow applies the selected Gemini interiority classifier to prepared
corpus passages without human gold labels or gold-label metrics. It is distinct
from `workflows/gemini_evaluation/`, which evaluates predictions against the
human annotation dataset.

The inference pipeline accepts any prepared passage-level corpus CSV with the
required fields. Its checked-in default is the frozen 29-book Armstrong/core
input at `data/frozen/corpus_inference/llm_input_armstrong.csv`; the matching
frozen predictions are
`data/frozen/corpus_inference/passage_predictions_armstrong.csv`. These frozen
files are not the complete 1,348-book corpus inference run.

See [INFERENCE_README.md](INFERENCE_README.md) for setup, runtime output names,
resume behavior, and command examples.
