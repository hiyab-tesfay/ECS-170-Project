# ECS 170 Project

Course project repository for ECS 170. The current code includes the Stage 1
machine-learning template and the Stage 2 training and test datasets.

## Project structure

```text
data/stage_1_data/          Stage 1 toy dataset
local_code/base_class/     Shared dataset, method, setting, result, and evaluation classes
local_code/stage_1_code/   Decision Tree, SVM, MLP, data loading, and evaluation code
result/stage_1_result/     Saved Stage 1 prediction results
script/stage_1_script/     Stage 1 executable scripts
stage_2_data/              Stage 2 instructions and CSV datasets
```

Stages 3 through 5 currently have package placeholders under `local_code/`.
Stages 1 and 2 have executable model code.

## Requirements

- Git
- [Git LFS](https://git-lfs.com/) — required for `stage_2_data/train.csv`
- Python 3.11 recommended
- `numpy`
- `scikit-learn`
- `torch`

The current code has been tested with:

```text
Python 3.11.15
NumPy 1.26.4
scikit-learn 1.9.1
PyTorch 2.2.2
```

PyCharm is optional, but it is the IDE used by the course setup instructions.

## Clone and install

Install Git LFS before cloning or pulling the repository:

```bash
git lfs install
git clone https://github.com/hiyab-tesfay/ECS-170-Project.git
cd ECS-170-Project
git lfs pull
```

Create and activate a virtual environment:

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import numpy, sklearn, torch; print(numpy.__version__, sklearn.__version__, torch.__version__)"
```

## Run Stage 1

The scripts use paths relative to `script/stage_1_script`, so that directory
must be the current working directory. The repository root must also be on
`PYTHONPATH` so Python can import `local_code`.

### macOS or Linux

```bash
cd script/stage_1_script
PYTHONPATH=../.. python script_decision_tree.py
PYTHONPATH=../.. python script_svm.py
PYTHONPATH=../.. python script_mlp.py
PYTHONPATH=../.. python script_load_result.py
```

### Windows PowerShell

```powershell
cd script/stage_1_script
$env:PYTHONPATH = "../.."
python script_decision_tree.py
python script_svm.py
python script_mlp.py
python script_load_result.py
```

The model scripts train and evaluate their respective models, then write binary
prediction files to `result/stage_1_result/`. Running them may overwrite existing
result files.

## PyCharm configuration

1. Open the repository as a PyCharm project.
2. Select the project's `.venv` as the Python interpreter.
3. Mark the repository root as **Sources Root**.
4. Set each Stage 1 run configuration's working directory to
   `script/stage_1_script`.
5. Run the four scripts listed above.

If Python reports `No module named local_code`, confirm that the repository root
is marked as a Sources Root or included in `PYTHONPATH`.

## Stage 2 data

`stage_2_data/train.csv` is stored with Git LFS because it exceeds GitHub's
normal per-file size limit. If the file contains a short text pointer instead of
CSV rows, install Git LFS and run:

```bash
git lfs pull
```

## Run Stage 2

From the repository root, activate the virtual environment and run:

```bash
source .venv/bin/activate
python script/stage_2_script/script_mlp.py
```

The script loads the supplied 60,000-row training split and 10,000-row test
split. It trains a `784 -> 128 -> 64 -> 10` MLP and a one-hidden-layer
`784 -> 128 -> 10` comparison using the same training settings. It reports test
accuracy, macro precision, macro recall, and macro F1 score. It saves
`metrics.json`, `loss_history.csv`, and `training_loss.svg` under
`result/stage_2_result/` for the project report.

The defaults are 10 epochs, batch size 256, Adam learning rate 0.001, and random
seed 42. Use `python script/stage_2_script/script_mlp.py --help` to see the
available settings.

## Result-file safety

Stage 1 results are serialized with Python `pickle`. Only load result files from
this repository or another trusted source, because untrusted pickle files can
execute arbitrary code when loaded.

## Run Stage 3 CNN experiments

Use Python 3.11 and install `requirements.txt`. Download the instructor's Stage 3
archive and extract `MNIST`, `ORL`, and `CIFAR` into `stage_3_data/` if those files
are absent. They are intentionally excluded from Git because of their size.
Only load these pickle files from a trusted source.

From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python script/stage_3_script/script_cnn.py --dataset all
```

Use `--dataset MNIST`, `ORL`, or `CIFAR` to run one dataset. The script runs a
baseline and two controlled ablations for each dataset. It holds out a fixed,
stratified 10% of the supplied training split for checkpoint selection; the
instructor's test split is used only for final evaluation. It writes metrics,
predictions, model weights, loss histories, and learning-curve PNGs under
`result/stage_3_result/<dataset>/`. Options include `--epochs`, `--patience`,
`--seed`, `--learning-rate`, and `--device auto|cpu|cuda|mps`.
Use `--variant baseline` to run just one configuration.
Single-variant runs write to a `single_<variant>/` subfolder so they do not
replace the complete comparison files.

The default epoch caps are 10 for MNIST, 35 for ORL, and 10 for CIFAR, with
early stopping after five epochs without validation-loss improvement. CPU runs,
especially CIFAR, can take substantial time. For a quick execution check, run
`--dataset ORL --epochs 1 --output-dir /tmp/stage3_smoke`.

### Optional device comparison

Run `python script/stage_3_script/benchmark_device.py` to time the same MNIST
baseline training batches on CPU and an available CUDA or MPS device. The script
records throughput and, when a GPU is available, its speedup in
`result/stage_3_result/device_benchmark.json`. If neither GPU backend is
available, it records that fact and a CPU timing. This short benchmark does not
replace the full test-set evaluations above.
