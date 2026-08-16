Pretrained GPU Training (RTX 4070)
==================================

This project now includes a GPU-first pretrained fine-tuning pipeline:
- train_pretrained.py
- predict_pretrained.py
- requirements_pretrained.txt

1) Install dependencies
-----------------------
pip install -r requirements_pretrained.txt

Use a CUDA-enabled PyTorch build. Verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

2) Train with pretrained model on GPU
-------------------------------------
PowerShell example:
$env:REQUIRE_CUDA="1"
$env:MODEL_NAME="microsoft/wavlm-base-plus"
$env:EPOCHS="30"
$env:BATCH_SIZE="8"
$env:EVAL_BATCH_SIZE="8"
$env:GRAD_ACCUM_STEPS="2"
$env:LR="1e-5"
$env:NUM_WORKERS="4"
python train_pretrained.py

Default data path:
.\data\RAVDESS

Outputs:
.\pretrained_outputs

3) Predict with trained model
-----------------------------
python predict_pretrained.py ".\data\RAVDESS\Actor_01\03-01-03-01-01-01-01.wav"

4) Generate evaluation report (poster-ready)
--------------------------------------------
python evaluate_pretrained.py

Outputs are written to:
.\pretrained_outputs\eval_report

Main files:
- evaluation_summary.json
- poster_results.txt
- classification_report.csv
- confusion_matrix.csv
- predictions.csv

5) Accuracy tips
----------------
- Keep REQUIRE_CUDA=1 so training fails fast if GPU is not active.
- Start with MODEL_NAME=microsoft/wavlm-base-plus (good quality/speed balance).
- If GPU memory allows, increase BATCH_SIZE and reduce GRAD_ACCUM_STEPS.
- Keep actor split for realistic generalization:
  $env:SPLIT_MODE="actor"
- Add more emotional speech datasets for larger gains than hyperparameter tuning alone.
