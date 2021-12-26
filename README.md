# pytorch_ddp_resnet
This repo contains a flexible implementation of deep residual networks. 
It supports the architectures and training algorithms from several papers, including:
- He et al., 2015 - ['Deep Residual Learning for Image Recognition'](https://arxiv.org/pdf/1512.03385.pdf), 
- He et al., 2016 - ['Identity Mappings in Deep Residual Networks'](https://arxiv.org/pdf/1603.05027.pdf),  
- Zagoruyko and Komodakis, 2016 - ['Wide Residual Networks'](https://arxiv.org/pdf/1605.07146.pdf).

## Getting started

We recommend creating a conda environment for this project. You can download the miniconda package manager from https://docs.conda.io/en/latest/miniconda.html.
Then you can create a new conda environment as follows:
```bash
conda create --name pytorch_ddp_resnet python=3.8.1
conda activate pytorch_ddp_resnet
git clone https://github.com/lucaslingle/pytorch_ddp_resnet
cd pytorch_ddp_resnet
pip install -e .
```

## Overview

This repo comes in two parts: a python package and a script. The script organizes all runs in a ```models_dir```, placing checkpoints and tensorboard logs in a ```run_name``` subdirectory. 

Furthermore, it expects to find a ```config.yaml``` file in the ```run_name``` directory, specifying hyperparameters and configuration details for the ```run_name``` training run. 

Using a flexible markup language like YAML allows us to specify, among other things, any pytorch optimizer and its keyword arguments, any pytorch learning rate scheduler and its keyword arguments, and any feasible image preprocessing/data augmentation logic. 

## Usage

To train a new model, you should thus:
- create a ```models_dir``` directory if it does not exist;
- create a subdirectory of the ```models_dir``` directory, with a descriptive name for the training run;
- copy over a config file and edit the parameters appropriately.

The config file has several components. We detail their usage below. 
- Distributed communication:
   - The script currently only supports single-machine training with zero or more GPUs. If using GPUs, be sure to set the config ```backend``` value to the string ```nccl```, and set the ```world_size``` to the number of GPUs available. 
   - Single-machine training is sufficient to train large ImageNet ResNet models, which typically require at most 8 GPUs. 
To scale up further, use your favorite cluster coordination tool, and be sure to set the ```master_hostname``` and ```master_port``` parameters to appropriate values.
- Data Augmentation:
  - We allow flexible pipeline of preprocessing and data augmentation operations to be specified. The transforms are defined in ```resnet/utils/transform_util.py```.
  - The class names for these transforms must be listed in the order they are to be applied.
  - The class names of each transform serves as a key in a dictionary, whose values are dictionaries of class-specific arguments.
- Architecure:
  - For the ```architecture_spec``` value, you should pick a string according to the documentation from ```ResNet``` class in ```resnet/architectures/resnet.py```.
- Optimizer and Scheduler:
  - For the optimizer and scheduler, you should use class names used by pytorch, with all capitalization intact.
  - For the scheduler, you should also set the ```scheduler_step_unit``` value to be ```epoch``` or ```batch```, depending on how your learning rate schedule is defined.
- Checkpointing:
  - Checkpointing can be performed by frequency or by performance (new best value).
  - To pick a checkpointing strategy, please set the ```checkpoint_strategy_cls_name``` to one of the CheckpointStrategy subclasses found in ```resnet/utils/checkpoint_util.py```.
  - We currently checkpoint at either the batch or epoch frequency, as specified by the ```checkpoint_strategy_args``` subargument ```unit```.

## Reproducing the Papers

To reproduce each paper, we use the exact same data augmentation as the paper, and evaluate exactly the same way.

| Paper           | Table   | Architecture |     Metric Name | Paper Result | Our Result | 
| --------------- | ------- | ------------ | --------------- | ------------ | ---------- | 
| He et al., 2015 | Table 6 |    ResNet-20 | Top-1 Error (%) |         8.75 |       8.19 | 
