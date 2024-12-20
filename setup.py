from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The LSTM-TD3 repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='lstm_td3',
    py_modules=['lstm_td3'],
    version='0.1',
    install_requires=[
        'joblib',
        'gym<=0.17.3',
        'numpy==1.26.4',
        'pybullet',
        'torch',
        'pandas',
        'gymnasium',
        'dm_control',
        'tensorboard',
        'ncps',
        'wandb'
    ],
    description="Long-Short-Term-Memory-based Twine Delayed Deep Deterministic Policy Gradient (LSTM-TD3)",
    author="Lingheng Meng",
)
