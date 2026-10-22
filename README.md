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

Stages 2 through 5 currently have package placeholders under `local_code/`, but
only Stage 1 has executable model code.

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

There is not yet a Stage 2 training script in the repository.

## Result-file safety

Stage 1 results are serialized with Python `pickle`. Only load result files from
this repository or another trusted source, because untrusted pickle files can
execute arbitrary code when loaded.
