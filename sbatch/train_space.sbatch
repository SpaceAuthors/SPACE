#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
set -euo pipefail
source "${SPACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/config.sh"
$SPACE_PYTHON - <<'PYEOF'
import torch; assert torch.cuda.is_available(), "CUDA is not available"
print("[CUDA]", torch.cuda.get_device_name(0))
PYEOF
$SPACE_PYTHON -u $SPACE_ROOT/train/train_space.py \
  --seed-list "${SEEDS:-0,1,2,3,4,5,6,7,8,9}" \
  --folds 5 --p1-epochs 10 --p2-epochs 12 --p3-epochs 150 --early-stop 40 \
  --target-active 512 --lam-damp 0.25 --fa-cap 0.107
