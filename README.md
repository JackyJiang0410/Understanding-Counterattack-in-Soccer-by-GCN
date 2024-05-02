# Understanding-Defensive-Performance-in-Soccer-by-GCN

This repository contains the code for training and evaluating a Graph Neural Network (GNN) using specific datasets and configurations.

## Overview

The script `run.ipynb` orchestrates the loading of data, model training, and evaluation. It utilizes modular components from `model_and_train`, `utils`, and `evaluation` modules to process the data, train the model, and compute evaluation metrics like ROC AUC and ECE.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` 
- `scikit-learn`
- `seaborn`
- `spektral`
- `livelossplot`
Install the required packages:
```bash
pip install -r requirements.txt
```

```bash
git clone https://github.com/JackyJiang0410/Understanding-Defensive-Performance-in-Soccer-by-GCN.git
cd Understanding-Defensive-Performance-in-Soccer-by-GCN
```

## Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/JackyJiang0410/Understanding-Defensive-Performance-in-Soccer-by-GCN.git
cd Understanding-Defensive-Performance-in-Soccer-by-GCN
```

## Usage
To run the model training and evaluation, use the following command:

```bash
python run.ipynb --model GNN --dataset combined --edge_feature 0 1 2 3 4 5 --node_feature 0 1 2 3 4 5 6 7 8 9 10 11 --matrix_type normal --learning_rate 0.00005 --epochs 300 --batch_size 16 --channels 128 --layers 3
```

You can customize the training by specifying additional command line arguments. Here are some key arguments you might consider:

- `--model`: Type of model to use (default GNN).
- `--dataset`: Dataset to train the model (default combined).
- `--edge_feature`: List of edge features (default [0, 1, 2, 3, 4, 5]).
- `--node_feature`: List of node features (default [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).
- `--matrix_type`: Type of matrix to use (default normal).
- `--learning_rate`: Initial learning rate (default 5e-5).
- `--epochs`: Number of epochs to train (default 300).
- `--batch_size`: Batch size for training (default 16).
- `--channels`: Number of channels (default 128).
- `--layers`: Number of layers (default 3).

### Custom Configuration
Modify the script's argument parser setup in `main.py` to include any additional or custom configurations specific to your needs.

## Output
The script will output the ROC AUC and ECE results and save the corresponding plots (roc_curve.png and calibration_curve.png) in the current directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
