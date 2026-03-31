#!/bin/bash

# 切换到 ubuntu 用户并激活 conda 环境
# 使用sudo -i登录用户,然后source激活脚本
# 使用--norc和--noprofile来避免加载.bashrc中的base激活
sudo -i -u ubuntu bash --norc -c 'source /workspace/code/C4/activate_env.sh && exec bash'
