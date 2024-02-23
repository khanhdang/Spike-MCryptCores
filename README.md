# Spike-MCryptCores
A light-weight neuromorphic controlling clock-gating based multi-core cryptography platform

## Setup

The project can be run on Google Collab.
For local Python enviroments, followings are the dependencies

- spikingjelly
- gdown (can be replaced by a local copy of dataset)
- onnxruntime
- numpy
- panda
- pytorch
- matplotlib

## Source code

[The Jupyter Notebook](aes_snn_flow.ipynb)


## Setting up

The source code only work with Python 3.7, therefore, we will use conda to manage.
Please install miniconda from [https://conda.io/](https://conda.io/).

Then, let's create a virtual environment

```
conda create --name py37 python=3.7
```

Then, let's activate the new virtual environment.

```
conda activate py37
```

Please check the version of Python with

```
python3 --version
```

After confirming it, please install the following packages:

```
pip3 install spikingjelly==0.0.0.0.8
pip3 install gdown
pip3 install onnxruntime
pip3 install pandas
pip3 install protobuf==3.17.3
```

To download the lastest files of data:

```
gdown --id 19GNGsv7x25WfQcOtWTmOOpDHK9fVQNNf&usp=drive_fs
gdown --id 19B2aNLO9IxIR4jqXERF4VEKN4OslC352&usp=drive_fs
gdown --id 1hJ_vMbVLauuS5i_sriJ8OXRbK1LZ8uc4
```

We also provide copies in this repo (data_training.csv, output.csv and data_testing-Full-random.csv).


To run the training and testing

```
python3 main.py
```

## Citation

- Pham-Khoi Dong, **Khanh N. Dang**,  Duy-Anh Nguyen, Xuan-Tu Tran, *''A light-weight neuromorphic controlling clockgating based multi-core cryptography platform'', **Microprocessors and Microsystems** (accepted), 2024. 	


## Contact

If you have any question, please contact

- Prof. Khanh N. Dang (khanh \[at\] u-aizu.ac.jp)
