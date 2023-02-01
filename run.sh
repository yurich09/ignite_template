#!/bin/bash

# 2 x Titan V (ddp + fp16)
python ignite_template/train.py data.loaders.batch=10
python ignite_template/train.py 'data.loaders={batch=10,maxscale=1.1,maxshift=0.05}'

python ignite_template/train.py data.loaders.batch=8
python ignite_template/train.py 'data.loaders={batch=8,maxscale=1.1,maxshift=0.05}'

# Titan V (cuda:1, fp16)
python ignite_template/train.py device=cuda:1 data.loaders.batch=4
python ignite_template/train.py device=cuda:1 'data.loaders={batch=4,maxscale=1.1,maxshift=0.05}'

python ignite_template/train.py device=cuda:1 data.loaders.batch=2
python ignite_template/train.py device=cuda:1 'data.loaders={batch=2,maxscale=1.1,maxshift=0.05}'

# 2 x 1080 Ti (ddp)
python ignite_template/train.py fp16=false data.loaders.batch=4
python ignite_template/train.py fp16=false 'data.loaders={batch=4,maxscale=1.1,maxshift=0.05}'

python ignite_template/train.py fp16=false data.loaders.batch=2
python ignite_template/train.py fp16=false 'data.loaders={batch=2,maxscale=1.1,maxshift=0.05}'
