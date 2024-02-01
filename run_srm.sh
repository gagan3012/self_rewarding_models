#!/bin/bash
#SBATCH --time=1-20:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --account=def-hasanc
#SBATCH --job-name=srm
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.11 scipy-stack gcc arrow cuda cudnn opencv StdEnv/2023 && source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

python sft_train.py /lustre07/scratch/gagan30/arocr/meta-llama/models/Mistral-7B-Instruct-v0.2

python rank_responses.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-sft 2

python reward_train.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-sft Mistral-7B-Instruct-v0.2-dpo-1 /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/Mistral-7B-Instruct-v0.2-sft.parquet

python rank_responses.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-dpo-1 3

python reward_train.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-dpo-1 Mistral-7B-Instruct-v0.2-dpo-2 /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/Mistral-7B-Instruct-v0.2-dpo-1.parquet

python rank_responses.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-dpo-2 4

python reward_train.py /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/Mistral-7B-Instruct-v0.2-dpo-2 Mistral-7B-Instruct-v0.2-dpo-3 /lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/Mistral-7B-Instruct-v0.2-dpo-2.parquet