#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=8000

module purge
module load TensorFlow SciPy-bundle scikit-learn

source $HOME/venvs/lfd_ass3/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

python3 --version
which python3

make lm

deactivate
