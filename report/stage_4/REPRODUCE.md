# Stage 4: recurrent text classification and generation

This implementation trains separate RNN, LSTM, and GRU networks for both tasks.
The instructor's classification task is IMDb binary movie-review sentiment; the
generation task is completion of short jokes from three starting words.

## Setup and input data

Use Python 3.11 and install the Stage 4 dependencies in the project virtual
environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stage4.txt
```

Download the instructor's `stage_4_data.zip` and 50-dimensional GloVe ZIP. Extract
the data so these paths exist:

- [Instructor Stage 4 dataset](https://drive.google.com/file/d/1yNr8deq1JO-HNhlAa1yV4Zycjqmr1Hfz/view)
- [Instructor GloVe 50d download](https://drive.google.com/file/d/1L9bkfMIyXG3xio2rQaothTU1XwGUcnY4/view)
- [Instructor report template](https://docs.google.com/document/d/1fLUhtUe_Rm4x-sL4N-LxBJ57k0enaCEnQNvWwTUkNY0/edit)

```text
stage_4_data/text_classification/train/neg/*.txt
stage_4_data/text_classification/train/pos/*.txt
stage_4_data/text_classification/test/neg/*.txt
stage_4_data/text_classification/test/pos/*.txt
stage_4_data/text_generation/data
```

`text_generation/data` is CSV with `ID,Joke` columns, not raw continuous text.
The GloVe ZIP can remain compressed; only matching vectors are retained in RAM.
The corpus and embedding ZIPs are excluded from Git.

Run all commands from the repository root:

```bash
python script/stage_4_script/prepare_data.py \
  --glove /path/to/glove.2024.wikigiga.50d.zip
python -m unittest discover -s tests -p 'test_stage4.py' -v
python script/stage_4_script/benchmark_device.py
```

The fixed seed is 42. Classification uses 22,500 training reviews, 2,500
stratified validation reviews, and the untouched 25,000 supplied test reviews.
Only training reviews determine the 20,000-entry vocabulary. HTML is removed,
case and apostrophes normalized, and punctuation and negation retained. Reviews
over 300 tokens retain the first 150 and last 150 tokens. PAD=0, UNK=1, BOS=2,
EOS=3. GloVe initializes non-special matching words; missing embeddings are
sampled from N(0, 0.05). Embeddings are trainable by default.

Generation deduplicates normalized complete jokes before a fixed 90/10 split.
No sliding windows cross jokes or split boundaries. Each input is BOS plus the
joke; each target is the same sequence shifted one word, ending in EOS. PAD is
excluded from token cross entropy. Its separate vocabulary is learned only from
the training jokes. The 96-token cap truncates no jokes in this seeded training
partition. No test set was supplied for generation; its validation results are
not an independent test estimate.

## Train the six required models

```bash
python script/stage_4_script/train.py --task classification --cell gru --device cpu
python script/stage_4_script/train.py --task classification --cell lstm --device mps
python script/stage_4_script/train.py --task classification --cell rnn --device cpu
python script/stage_4_script/train.py --task generation --cell all --device cpu
```

These are the measured classifier devices. On a machine without MPS, replace
`--device mps` with `--device cpu`; the entire comparison can also run with
`--task classification --cell all --device cpu`. Device-dependent numerical
differences can change the exact results. The benchmark supports the reported
device choices but is not a controlled device-accuracy study.

Classifiers use 50-dimensional GloVe, forward and reverse 128-unit recurrent
encoders, masked mean/max pooling, dropout 0.4, and two logits. Each direction
reverses only actual words, allowing efficient dense kernels with correct
padding behavior. Tests compare this implementation against PyTorch's packed
bidirectional reference. For multiple layers, each direction has an independent
stack; the two directions are concatenated at the top. The comparison uses one
layer throughout. Adam uses lr=0.001, batch size 64, weight decay 1e-5, gradient
clip norm 1, at most 12 epochs, and early stopping after four epochs without
validation-loss improvement. The learning rate halves after three unimproved
epochs. The final test evaluation uses the minimum-validation-loss checkpoint.

Generators are causal, one-layer 256-unit recurrent networks with the same
50-dimensional trainable GloVe initialization, dropout 0.2, and vocabulary-wide
output logits. Adam starts at lr=0.003, with cosine decay to 0.0003 over 100
epochs, batch size 64, weight decay 1e-5 and gradient clip norm 1. Both the
minimum-validation-loss and final-epoch checkpoints and samples are retained.
Later checkpoints can produce more recognizable training-style completions
while overfitting; the report distinguishes this from held-out generalization.

Initialization uses Xavier-uniform input/output weights, gate-wise orthogonal
recurrent weights, and zero biases. Python, NumPy, and PyTorch seeds are fixed;
cross-device bitwise equality is not promised. Use `--help` for all settings.
Completed experiments are protected against overwriting; choose a different
`--run NAME` for a new experiment. `--skip-completed` skips completed models.
To train with another data split, prepare a separate cache using `--seed` and
pass that same seed to training. Checkpoints and caches use the restricted
PyTorch tensor loader (`weights_only=True`).

## Extra controlled ablation

The additional classification experiment removes GloVe while holding the model,
split and training settings fixed:

```bash
python script/stage_4_script/train.py --task classification --cell gru \
  --device cpu --random-embeddings --run random_embeddings
```

The random ablation uses PyTorch Embedding's N(0, 1) initialization with a zero
PAD row; this replaces the entire pretrained matrix. It is not a frozen-vector
ablation.

The RNN/LSTM/GRU comparison itself is the required recurrent-unit ablation. The
generation examples also compare sampling temperatures while holding each
checkpoint, prompt and seed fixed.

## Use trained models

```bash
python script/stage_4_script/classify.py \
  --checkpoint result/stage_4_result/classification/comparison/gru/best.pt \
  --device cpu --text 'The acting was excellent and I loved the story.'

python script/stage_4_script/generate.py \
  --checkpoint result/stage_4_result/generation/comparison/lstm/last.pt \
  --device cpu --prompt 'what did the' --temperature 0.5 --seed 42
```

Generation accepts exactly three whitespace-separated starting words. Inference
loads only model weights and the saved vocabulary; it never looks up a training
joke. PAD, UNK and BOS are suppressed at decoding. Generation stops at EOS or
the configured new-token cap, with a mild repetition penalty and top-k sampling.
Unknown prompt words are reported. Each model's saved sample sweep uses six
prompts, two temperatures and three seeds. Similarity comparisons to training
jokes are performed after decoding, exclusively for evaluation. Exact matches
must be interpreted as memorization, not proof of new joke construction.

## Saved evidence and report

All evidence is under `result/stage_4_result/`:

- Dataset statistics, GloVe SHA-256/coverage, and exact split manifests.
- `classification/comparison/{rnn,lstm,gru}/`: configuration, vocabulary,
  history, best/final weights, 25,000 predictions, metrics, confusion matrix,
  and Wilson 95% accuracy intervals.
- `generation/comparison/{rnn,lstm,gru}/`: configuration, vocabulary,
  history, best/final weights, validation perplexity, and both sample sweeps.
- `device_benchmark.json`: synchronized CPU/GPU timings after warm-up. This is
  a short throughput comparison, not a full device accuracy comparison.
- `figures/`: architecture diagrams, learning curves, and confusion matrices.

Generate the figures and report after training:

```bash
python script/stage_4_script/plot_results.py
python script/stage_4_script/finalize_metadata.py
python script/stage_4_script/build_report.py
python script/stage_4_script/check_submission.py
```

The report follows the instructor's sections, 10-point body text, single
spacing, and the five-page maximum. Personal/team details are intentionally
omitted as requested. Model weights and preprocessing caches are excluded from
Git because of size; the local submission ZIP includes reusable model weights.
The source repository contains the code and measured evidence needed to
reproduce training. The local archive also supports immediate inference without
retraining; keep it with the submission.
