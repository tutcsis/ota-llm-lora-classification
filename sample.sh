#PBS -q gLrchq
#PBS -l select=1:ncpus=8:mem=64G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.31.0
#PBS -k doe -j oe

cd ${PBS_O_WORKDIR}
poetry run accelerate launch --mixed_precision=bf16 src/sample.py --model_name rinna/japanese-gpt-neox-3.6b