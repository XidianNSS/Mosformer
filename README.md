# Mosformer: Maliciously Secure Three-Party Inference Framework for Large Transformers

A maliciously secure framework for efficient 3-party protocols tailored for  Transformer model inference.

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
git clone --recurse-submodules https://github.com/jxxyh/Mosformer.git
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
| `rss`       | Evaluation of replicated secret sharing (RSS)-based secure computation     |
| `bench`     | Microbenchmarking of core secure operations (e.g., ReLU, MatMul, Softmax)  |
| `cnn3pc`    | Secure inference for CNN models, including AlexNet and ResNet50            |
| `llm3pc`    | Secure inference for Transformer models: Vanilla Transformer, BERT, GPT2   |
| `llmacc`    | Accuracy evaluation for BERT-Base and GPT2 models                          |

> **Note:** For `llmacc`, make sure to place the model and dataset shares in the following directories:
> - `./log/model_shares/`
> - `./log/data_shares/`
