# Understanding-Counterattack-in-Soccer-by-GCN

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

### Parameters

- `--model`: Type of model to use (default `GNN`). Choose a GNN-based model suitable for graph-based data processing.

- `--dataset`: Select dataset for training the model. Options are:
  - `men`: MLS 2022 season data.
  - `women`: NWSL 2022 season + International Women's soccer.
  - `combined`: Combination of both men's and women's data.

- `--edge_feature`: List of edge features to include (integer indices). Options include:
  - `0`: Player Distance - Distance between two players connected to each other.
  - `1`: Speed Difference - Speed difference between two players connected to each other.
  - `2`: Positional Sine angle - Sine of the angle created between two players in the edge.
  - `3`: Positional Cosine angle - Cosine of the angle created between two players in the edge.
  - `4`: Velocity Sine angle - Sine of the angle created between the velocity vectors of two players in the edge.
  - `5`: Velocity Cosine angle - Cosine of the angle created between the velocity vectors of two players in the edge.

- `--node_feature`: List of node features to include (integer indices). Options include:
  - `0`: x coordinate - x coordinate on the 2D pitch for the player/ball.
  - `1`: y coordinate - y coordinate on the 2D pitch for the player/ball.
  - `2`: vx - Velocity vector's x coordinate.
  - `3`: vy - Velocity vector's y coordinate.
  - `4`: Velocity - magnitude of the velocity.
  - `5`: Velocity Angle - angle made by the velocity vector.
  - `6`: Distance to Goal - distance of the player from the goal post.
  - `7`: Angle with Goal - angle made by the player with the goal.
  - `8`: Distance to Ball - distance from the ball (always 0 for the ball).
  - `9`: Angle with Ball - angle made with the ball (always 0 for the ball).
  - `10`: Attacking Team Flag - 1 if the team is attacking, 0 if not and for the ball.
  - `11`: Potential Receiver - 1 if player is a potential receiver, 0 otherwise.

- `--matrix_type`: Type of adjacency matrix to use. Options are:
  - `normal`: Connects attacking and defensive players through different schemes.
  - `delaunay`: Connects players in a delaunay matrix fashion.
  - `dense`: Connects all players and the ball to each other.
  - `dense_ap`: Connects all attacking players to each other and defensive players.
  - `dense_dp`: Connects all defending players to each other and attacking players.

- `--learning_rate`: Initial learning rate (default `5e-5`).

- `--epochs`: Number of epochs to train (default `300`).

- `--batch_size`: Batch size for training (default `16`).

- `--channels`: Number of channels (default `128`).

- `--layers`: Number of layers (default `3`).

### Custom Configuration
Modify the script's argument parser setup in `main.py` to include any additional or custom configurations specific to your needs.

## Output
The script will output the ROC AUC and ECE results and save the corresponding plots (roc_curve.png and calibration_curve.png) in the current directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
