#PBS -q gLiotq
#PBS -l select=1:ncpus=8:mem=256G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2
#PBS -k doe -j oe

cd ${PBS_O_WORKDIR}
#export TORCH_USE_CUDA_DSA=1
#export CUDA_LAUNCH_BLOCKING=1
#poetry run accelerate launch --mixed_precision=bf16 src/train.py --model_name llm-jp/llm-jp-3-3.7b-instruct
poetry run accelerate launch --mixed_precision=bf16 src/train.py --model_name tokyotech-llm/Swallow-7b-hf