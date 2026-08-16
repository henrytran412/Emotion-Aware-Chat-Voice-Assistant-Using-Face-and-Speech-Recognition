$ErrorActionPreference = "Stop"

$PythonExe = "C:/Users/jacob/eng_success/dataset_CUDA/.venv/Scripts/python.exe"

$env:REQUIRE_CUDA = "1"
$env:MODEL_NAME = "microsoft/wavlm-base-plus"
$env:SPLIT_MODE = "random"
$env:SEED = "42"

# Tuned defaults to reduce class-collapse on happy/neutral.
$env:EPOCHS = "35"
$env:BATCH_SIZE = "4"
$env:EVAL_BATCH_SIZE = "4"
$env:GRAD_ACCUM_STEPS = "4"
$env:LR = "1e-5"
$env:WEIGHT_DECAY = "0.01"
$env:WARMUP_RATIO = "0.08"
$env:NUM_WORKERS = "0"

$env:USE_ATTENTION_MASK = "1"
$env:USE_CLASS_WEIGHTS = "1"
$env:BALANCED_SAMPLING = "1"
$env:LABEL_SMOOTHING = "0.05"
$env:BEST_METRIC = "eval_weighted_f1"
$env:EARLY_STOP_PATIENCE = "6"

$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"
$env:OUTPUT_DIR = "C:\Users\jacob\eng_success\speech_recognition\speech_emotion_project1\pretrained_outputs_tuned"

& $PythonExe train_pretrained.py
if ($LASTEXITCODE -ne 0) { throw "train_pretrained.py failed with exit code $LASTEXITCODE" }

$env:MODEL_DIR = $env:OUTPUT_DIR
$env:EVAL_OUT_DIR = "$env:OUTPUT_DIR\eval_report"
& $PythonExe evaluate_pretrained.py
if ($LASTEXITCODE -ne 0) { throw "evaluate_pretrained.py failed with exit code $LASTEXITCODE" }

& $PythonExe create_poster_graphs.py --model-dir $env:OUTPUT_DIR
if ($LASTEXITCODE -ne 0) { throw "create_poster_graphs.py failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Done. Results in: $env:OUTPUT_DIR"
Write-Host "Evaluation report: $env:EVAL_OUT_DIR"
Write-Host "Poster graphs: $env:EVAL_OUT_DIR\poster_graphs"
