#PBS -q gLiotq
#PBS -l select=1:ncpus=4:mem=128G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2
#PBS -k doe -j oe

cd ${PBS_O_WORKDIR}
TORCH_HOME=`pwd`/.cache/torch
TRANSFORMERS_CACHE=`pwd`/.cache/transformers
HF_HOME=`pwd`/.cache/huggingface
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME
export TORCH_USE_CUDA_DSA=1
#export TORCH_USE_CUDA_DSA=1
#export CUDA_LAUNCH_BLOCKING=1
# poetry run accelerate launch --mixed_precision=bf16 src/train.py --model_name rinna/bilingual-gpt-neox-4b
# poetry run accelerate launch --mixed_precision=bf16 src/train.py --model_name llm-jp/llm-jp-3-13b
poetry run accelerate launch --mixed_precision=bf16 src/train.py --model_name tokyotech-llm/Swallow-13b-hf

