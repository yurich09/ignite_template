#!/bin/bash

# 2 x Titan V (ddp + fp16)
python ignite_template/train.py -m data.loaders.batch=10,8
python ignite_template/train.py -m data.loaders.batch=10,8 'data.prepare={maxscale:1.1,maxshift:0.05}'

# Titan V (cuda:1, fp16)
python ignite_template/train.py -m device=cuda:1 data.loaders.batch=4,2
python ignite_template/train.py -m device=cuda:1 data.loaders.batch=4,2 'data.prepare={maxscale:1.1,maxshift:0.05}'

# 2 x 1080 Ti (ddp)
python ignite_template/train.py -m fp16=false data.loaders.batch=4,2
python ignite_template/train.py -m fp16=false data.loaders.batch=4,2 'data.prepare={maxscale:1.1,maxshift:0.05}'
