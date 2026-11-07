#!/usr/bin/env bash
set -e
if [ ! -x "$HOME/.pixi/bin/pixi" ]; then curl -fsSL https://pixi.sh/install.sh | bash; fi
export PATH="$HOME/.pixi/bin:$PATH"
mkdir -p "$HOME/csds_gpu"
cd "$HOME/csds_gpu"
if [ ! -f pixi.toml ]; then pixi init --channel conda-forge --platform linux-64; fi
pixi add "python=3.11" polars pyarrow scikit-learn numpy scipy
pixi add --pypi "xgboost-cu12==3.2.0"
pixi run python -c "import xgboost as xgb, polars; print('xgboost', xgb.__version__, 'CUDA', xgb.build_info().get('USE_CUDA'), 'polars', polars.__version__)"