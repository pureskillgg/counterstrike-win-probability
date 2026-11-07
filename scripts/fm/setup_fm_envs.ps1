# Isolated Python environments for the foundation-model test (nothing is installed into the project's own environments).
#   G:\pureskill\envs\mitra   : AutoGluon tabular with the Mitra model (Apache-2.0 weights, ~76M parameters) + PyTorch (CUDA 12.6 build)
#   G:\pureskill\envs\tabicl  : TabICL (BSD-3-Clause code; checkpoints download from Hugging Face on first use) + PyTorch
# Only the TabPFN-2.5-and-later weights and LimiX-2 are left out: their licences forbid commercial use (see notes/research/RESEARCH_ROUND4).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fm\setup_fm_envs.ps1
$ErrorActionPreference = 'Continue'
$envs = 'G:\pureskill\envs'
New-Item -ItemType Directory -Force $envs | Out-Null
$torch = 'https://download.pytorch.org/whl/cu126'
foreach ($e in @(@{ name = 'mitra'; pkgs = @('autogluon.tabular[mitra]') }, @{ name = 'tabicl'; pkgs = @('tabicl') })) {
  $dir = "$envs\$($e.name)"
  $py = "$dir\Scripts\python.exe"
  "$(Get-Date -Format s) === $($e.name)"
  if (-not (Test-Path $py)) { uv venv $dir --python 3.12 }
  uv pip install --python $py torch --index-url $torch
  uv pip install --python $py @($e.pkgs) polars pyarrow numpy scikit-learn scipy
  & $py -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
  "$(Get-Date -Format s) === $($e.name) done"
}
