

# Learning Versatile Skills with Curriculum Masking

This codebase is the official implementation of [CurrMask](https://arxiv.org/abs/2410.17744).

## Get Started
### Environments

Install [MuJoCo](http://www.mujoco.org/):

* Download MuJoCo binaries [here](https://mujoco.org/download).
* Unzip the downloaded archive into `~/.mujoco/`.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.


Install dependencies in conda environment:
```sh
conda env create -f environment.yaml
conda activate currmask
```

### Collect data

You can follow the example scripts in ``data_collection/scripts`` and collect offline data as described in our paper. You can also collect your own dataset following the instructions:
```
conda activate currmask
cd data_collection
bash scripts/sup.sh             #To collect supervised data
bash scripts/unsup.sh           #To collect unsupervised data
```

### Train and Eval

We provide example scritps in folder ``scripts`` to pretrain or evaluate the model with skill prompting, goal-conditioned planning and offline RL. An example is:
```
bash scripts/pretrain.sh
bash scripts/eval.sh
```
## Citation
If you find our work helpful, please kindly cite as
```
@article{tang2024currmask,
      title={Learning Versatile Skills with Curriculum Masking}, 
      author={Yao Tang and Zhihui Xie and Zichuan Lin and Deheng Ye and Shuai Li},
      journal={arXiv preprint arXiv:2410.17744},
      year={2024},
      url={https://arxiv.org/abs/2410.17744}, 
}
```

## Acknowledgements
This code is built on [MaskDP](https://github.com/FangchenLiu/MaskDP_public). We would like to express our gratitude to the authors for open-sourcing code to the community!