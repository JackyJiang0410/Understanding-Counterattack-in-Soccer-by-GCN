# -*- coding: utf-8 -*-
"""gnn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WKKmIjnWgVwNX_2v62PGvQD4SyEhPEzi
"""

import numpy as np
import tensorflow as tf
from spektral.data import Dataset, Graph, DisjointLoader
from spektral.layers import GCNConv, GlobalAvgPool, CrystalConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from livelossplot import PlotLosses

class CounterDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs['data']
        self.matrix_type = kwargs['matrix_type']

    def read(self):
        if self.matrix_type not in self.data:
            raise ValueError(f"Matrix type {self.matrix_type} not found in data.")
        data = self.data
        data_mat = data[self.matrix_type]
        graphs = [
            Graph(x=x, a=a, e=e, y=y) for x, a, e, y in zip(
                data_mat['x'], data_mat['a'], data_mat['e'], data['binary']
            )
        ]
        if not graphs:
            raise ValueError("No graphs could be loaded from the provided data.")

        return graphs

class GCNModel(Model):
    def __init__(self, n_out, n_layers, channels, final_activation='sigmoid'):
        super().__init__()
        self.gcn_layers = [GCNConv(channels, activation=LeakyReLU(alpha=0.01)) for _ in range(n_layers)]
        self.pool = GlobalAvgPool()
        self.dense_layers = [
            Dense(channels, activation=LeakyReLU(alpha=0.01)),
            Dropout(0.5),
            Dense(n_out, activation=final_activation)
        ]

    def call(self, inputs):
        x, a, e, i = inputs
        for gcn in self.gcn_layers:
            x = gcn([x, a])
        x = self.pool([x, i])
        for dense in self.dense_layers:
            x = dense(x)
        return x

class GNN(Model):
    def __init__(self, n_out, n_layers, channels,  activation='relu', final_activation='sigmoid'):
        super().__init__()
        self.layers_list = [CrystalConv() for _ in range(n_layers)]
        self.pool = GlobalAvgPool()
        self.dense_layers = [
            Dense(channels, activation=activation),
            Dropout(0.5),
            Dense(channels, activation=activation),
            Dropout(0.5),
            Dense(n_out, activation=final_activation)
        ]

    def call(self, inputs):
        x, a, e, i = inputs
        for layer in self.layers_list:
            x = layer([x, a, e])
        x = self.pool([x, i])
        for dense in self.dense_layers:
            if isinstance(dense, Dropout):
                x = dense(x, training=True)
            else:
                x = dense(x)
        return x

def create_and_train_model(data, matrix_type, model_type, learning_rate, epochs, batch_size, channels, layers):
  dataset = CounterDataset(data = data, matrix_type = matrix_type)

  N = max(g.n_nodes for g in dataset) # Number of nodes
  F = dataset.n_node_features  # Dimension of node features
  S = dataset.n_edge_features  # Dimension of edge features
  n_out = dataset.n_labels  # Dimension of the target
  n = len(dataset) # Number of samples in the dataset

  # Train/test split for the dataset
  idxs = np.random.RandomState(seed=15).permutation(len(dataset))
  split_va, split_te = int(0.7 * len(dataset)), int(0.69 * len(dataset))
  idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
  dataset_tr = dataset[idx_tr]
  dataset_te = dataset[idx_te]
 # When creating the loader, check if you need to adjust batch format
  loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
  loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)

  if model_type == 'GNN':
    model = model = GNN(n_out, n_layers=layers, channels=channels)  # Adjust parameters as necessary
  else:
    model = GCNModel(n_out, n_layers=layers, channels=channels)  # Adjust parameters as necessary
  optimizer = Nadam(learning_rate=learning_rate)
  loss_fn = BinaryCrossentropy()

  # Initialize the live loss plot instance
  live_plot = PlotLosses()

  # Initialize tracking variables
  step = 0
  cumulative_loss = 0

  @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
  def train_step(inputs, target):
      with tf.GradientTape() as tape:
          predictions = model(inputs, training=True)  # Pass inputs directly
          loss = loss_fn(target, predictions) + sum(model.losses)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return loss




  # Print loss at each step of training.
  step = loss = 0
  for batch in loader_tr:
      step += 1
      loss += train_step(*batch)
      if step == loader_tr.steps_per_epoch:
          step = 0
          print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
          loss = 0

  return loader_tr, loader_te, model