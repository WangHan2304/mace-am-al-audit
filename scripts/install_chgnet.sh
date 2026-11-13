#!/bin/bash
set -e
PIP="$HOME/miniconda3/envs/chgnet/bin/pip"
echo "step1 torch"
"$PIP" install --no-input torch --index-url https://download.pytorch.org/whl/cpu
echo "step2 chgnet"
"$PIP" install --no-input chgnet ase
echo "step3 verify"
"$HOME/miniconda3/envs/chgnet/bin/python" -c "import torch, chgnet, ase; print('OK', torch.__version__, ase.__version__)"
echo ALL_DONE
