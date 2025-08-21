# Mosformer: Maliciously Secure Three-Party Inference Framework for Large Transformers

A maliciously secure framework for efficient 3-party protocols tailored for  Transformer model inference. This work have been accepted by ACM CCS 2025.

## Note
This is an academic proof-of-concept prototype and is still under development.
It **should not** be used in any security sensitive product.

This repository currently provides the online phase implementation for privacy-preserving inference of large language models.
The offline phase was implemented with [NssMPClib](https://github.com/XidianNSS/NssMPClib), and the performance results in our paper are based on that implementation and derived computations rather than being fully implemented here.

## Requirements

- The code should work on most Linux distributions and it has been developed and tested with Ubuntu 24.04.

- Requirements
    - g++ (C++17 compatible)
    - CMake ($\ge$ 3.16)
    - make
    - OpenMP
    - [cnpy](https://github.com/rogersce/cnpy)

## Installation and Build

### Install Dependencies

```bash
sudo apt update
sudo apt install build-essential
```

### Clone the Repository

```bash
git clone --recurse-submodules https://github.com/XidianNSS/Mosformer.git
```

OR

```bash
git clone https://github.com/XidianNSS/Mosformer.git
git submodule update --init --recursive
```

### Build and Run

1. Manual Build and Run
```bash
mkdir build
cd ./build
cmake ..
make -j
./mosformer [test_name] [party_id]
```

2. Use Provided Scripts
```bash
cd ./scripts
sh eval_bash.sh [test_name] [(optional)party_id]
```

3. Only build
```bash
mkdir build
cd ./build
cmake ..
make -j
```

OR

```bash
cd ./scripts
sh eval_bash.sh
```

- `test_name`: Choose one of the supported tests (see below).

- `party_id`: Choose 0, 1, or 2 to run as one party. Leave blank to run all three. parties locally

### Supported `test_name` Options

| `test_name` | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `-h, --help`| Show this help message and exit                                            |
| `rss`       | Evaluation of replicated secret sharing (RSS)-based secure computation     |
| `bench`     | Microbenchmarking of core secure operations (e.g., ReLU, MatMul, Softmax)  |
| `cnn3pc`    | Secure inference for CNN models, including AlexNet and ResNet50            |
| `llm3pc`    | Secure inference for Transformer models: Vanilla Transformer, BERT, GPT2   |
| `llmacc`    | Accuracy evaluation for BERT-Base and GPT2 models                          |

> **Note:** For `llmacc`, make sure to place the model and dataset shares in the following directories:
> - `./log/model_shares/`
> - `./log/data_shares/`  

#### How to get shares

We provide a Python script `./tests/pt2npz.py` to help convert and share plaintext model parameters and input data. This script relies on [NssMPClib](https://github.com/XidianNSS/NssMPClib) — please follow its documentation to configure the required environment.  

Before running the script, please prepare the plaintext model parameter files and input data files:  

- **Model parameters** should be named as `[model_name]_[dataset_name].pt` and stored in `./log/model_save/`.  
- **Input data** should be named as `[dataset_name].pt` and stored in `./log/data_save/`. The input data should be the **embedding hidden states**.  

Currently, the following model–dataset pairs are supported by default:  

- `Bert_base` → `RTE`, `QNLI`, `STS-B`  
- `GPT2` → `WikiText103`  

You may add new models and datasets as needed, but make sure to also update the corresponding files, such as `./tests/pt2npz.py` and `./tests/llm_acc.cpp`.  

Once preparation is complete, you can share the model and data with:  

```bash
cd ./tests
python pt2npz.py [model_name] [dataset]
```
The shared model and data will be stored in `./log/model_shares/` and `./log/data_shares/`.

For help information, run:

```bash
python pt2npz.py -h
```


## Contact us
Email: yuhengxia@stu.xidian.edu.cn


## Citation
```
@inproceedings{cheng2025mosformer,
  title     = {Mosformer: Maliciously Secure Three-Party Inference Framework for Large Transformers},
  author    = {Ke Cheng and Yuheng Xia and Anxiao Song and Jiaxuan Fu and Wenjie Qu and Yulong Shen and Jiaheng Zhang},
  booktitle = {32nd ACM Conference on Computer and Communications Security, CCS 2025},
  year      = {2025},
  month     = {October},
  address   = {Taipei, Taiwan, China}
}
```