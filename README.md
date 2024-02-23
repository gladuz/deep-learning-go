# Deep learning playground scripts in Go

## Overview
- `/tensor` Tensor implementation from scratch. Implementation of small Tensor model taken from [Pytorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/). Supports `At(x,y)`, `View(...idx)` with arbitrary sizes.
- `/simple_transformer` I've built a Transformer inference code from scratch in Go, without relying on external dependencies. The inspiration? Well, let's just say [minGPT](https://github.com/karpathy/minGPT) had a hand in it 😉. But heads up, there might be a few bugs as it was a weekend project without too much attention (no pun intended) to detail.

## Features per folder
- transformer
    - Train a Transformer model in PyTorch on Iris dataset.
    - Export the model weights.
    - Perform inference in Go without external dependencies.
- tensor
    - Generic `Tensor` implementation with arbitrary sizes
    - Testable code for `At, View, DimSlice` with the same underlying storage (like Pytorch)
## Requirements

To run this project, you'll need:

- Python 3.x
- PyTorch and Scikit-learn
- Go compiler

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your_username/transformer-inference-go.git
    ```

2. Install Python dependencies (only Pytorch and Scikit-learn):

    ```bash
    pip install -r requirements.txt
    ```


## Training the Transformer Model

1. Navigate to the main directory:

2. Train the Transformer model using PyTorch. Modify the training script (`transformer.py`) to suit your dataset

```bash
python transformer.py
```
Model weights will be written into the `weights.bin` file
### Inference in GO
1. Run the Go file:
```bash
go run simple_attn.go
```
It will read the dataset, run prediction on samples and calculate the accuracy

## Tensor implementation
Tests are in `tensor/tensor_test.go` file. The correct results are taken from the pytorch implementation.