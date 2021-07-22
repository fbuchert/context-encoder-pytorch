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