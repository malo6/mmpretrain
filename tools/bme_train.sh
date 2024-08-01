#!/usr/bin/env bash
#SBATCH -p bme_gpu4
#SBATCH --job-name=selfsup_reflacx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 5-00:00:00

set -x

CONFIG=$1
PY_ARGS=${@:2}
source activate mmlab
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=4 python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
# gpu:1
# gpu:NVIDIAA10080GBPCIe:1#