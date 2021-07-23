# PyTorch Implementation: Context Encoder
PyTorch implementation of [Context Encoders: Feature Learning by Inpainting
](https://arxiv.org/abs/1604.07379) based on the [official lua implementation](https://github.com/pathak22/context-encoder).

The implementation supports the following datasets:
- CIFAR-10 / CIFAR-100
- SVHN
- Caltech101 / Caltech256
- STL10
- HAM10000
- ImageNet


## Installation
Required python packages are listed in `requirements.txt`. All dependencies can be installed using pip
```
pip install -r requirements.txt
```
or using conda
```
conda install --file requirements.txt
```

## Training
Context encoder training is started by running the following command:
```
python main.py
```
All commandline arguments, which can be used to adapt the configuration of the context encoder are defined and described in `arguments.py`.
By default the following configuration is run:
```
dataset: 'cifar10'
epochs: 50
batch_size: 64
lr: 0.0002
beta1: 0.5
beta2: 0.999
w_rec: 0.999
overlap: 0
bottleneck: 4000
image_size: 32
mask_area: 0.25
device: 'cuda'
out_dir: 'context_encoder'
```
In addition to these, the following arguments can be used to further configure the context encoder training process:
* `--device <cuda / cpu>`: Specify whether training should be run on GPU (if available) or CPU
* `--num-workers <num_workers>`: Number of workers used by torch dataloader
* `--resume <path to run_folder>`: Resumes training of training run saved at specified path, e.g. `'out/context_encoder_training/run_0'`. Dataset splits, model state, optimizer state, etc.
  are loaded and training is resumed with specified arguments.
* see `arguments.py` for more

Alternatively, the `polyaxon.yaml`-file can be used to start the context encoder training on a polyaxon-cluster:
```
polyaxon run -f polyaxon.yaml -u
```
For a general introduction to polyaxon and its commandline client, please refer to the [official documentation](https://github.com/polyaxon/polyaxon)
## Monitoring
The training progress (loss, accuracy, etc.) can be monitored using tensorboard as follows:
```
tensorboard --logdir <result_folder>
```
This starts a tensorboard instance at `localhost:6006`, which can be opened in any common browser.

## Evaluation


## References
```
@inproceedings{pathak2016context,
  title={Context encoders: Feature learning by inpainting},
  author={Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2536--2544},
  year={2016}
}
```