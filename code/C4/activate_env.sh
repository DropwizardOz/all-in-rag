#!/bin/bash
# 在ubuntu用户下激活conda环境并切换到工作目录

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda deactivate  # 先取消激活
conda activate all-in-rag
cd /workspace/code/C4
echo "Conda环境已激活: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python)"
