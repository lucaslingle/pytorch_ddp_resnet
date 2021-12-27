from setuptools import setup


setup(
    name="pytorch_ddp_resnet",
    py_modules=["pytorch_ddp_resnet"],
    version="0.0.1",
    description="A Pytorch implementation of Deep Residual Networks.",
    author="Lucas D. Lingle",
    install_requires=[
        'torch==1.10.1',
        'torchvision==0.11.2',
        'tensorboard==2.7.0',
        'pyyaml==6.0',
        'filelock==3.4.0',
        'pillow==8.1.2'
    ]
)