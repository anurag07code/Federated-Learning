# Federated Learning Labs

This repository contains a series of Jupyter notebooks to learn about Federated Learning (FL) using the Flower framework.

## Labs

### LAB01: Why Federated Learning?

This lab introduces the challenges of training machine learning models on data that is distributed and non-identically distributed (non-IID).

*   `L1_Why_Federated_Learning.ipynb`: Demonstrates how models trained on isolated, non-IID datasets perform poorly on a global test set. It uses the MNIST dataset, partitioned among three clients, where each client has only a subset of the digits.
*   `utils1.py`: Contains helper functions for data loading, model definition (`SimpleModel`), training, and evaluation.
*   `requirements.txt`: Lists the necessary Python packages for this lab.

### LAB02: Federated Learning Training Process

This lab shows how to implement a basic federated learning system with Flower to overcome the challenges identified in LAB01.

*   `L2_Federated_Learning_Training_Process.ipynb`: Implements a Flower simulation with a server and three clients. Each client trains a model on its non-IID data partition, and the server aggregates the model updates using the Federated Averaging (FedAvg) strategy. The resulting global model shows improved performance.
*   `utils2.py`: Provides utility functions for the Flower simulation, including the client and server logic.
*   `requirements.txt`: Lists the necessary Python packages.

### LAB03: Tuning Federated Learning

This lab explores how to fine-tune the federated learning process.

*   `L3_Tuning.ipynb`: Demonstrates how to dynamically adjust training parameters, such as the number of local training epochs on the clients, during the federated learning process using Flower's `on_fit_config_fn`. This allows for more sophisticated training strategies.
*   `utils3.py`: Contains utility functions for this lab, including more advanced helpers for the Flower simulation.
*   `requirements.txt`: Lists the necessary Python packages.

## Use Cases

This repository also includes several use cases that demonstrate how to apply federated learning to different tasks using both Flower and TensorFlow Federated (TFF).

### Use Case 1: Image Classification

*   `usecases/UC1_Flower.ipynb`: Implements federated image classification on the MNIST dataset using Flower.
*   `usecases/UC1_TFF.ipynb`: Implements the same image classification task using TensorFlow Federated.

### Use Case 2: Sentiment Analysis

*   `usecases/UC2_Flower.ipynb`: Demonstrates federated sentiment analysis on the Sentiment140 dataset using Flower. It includes text preprocessing and training a model with a pre-trained embedding layer.

### Use Case 5: Clustering

*   `usecases/UC5_TFF.ipynb`: Shows how to perform federated clustering using the k-means algorithm with TensorFlow Federated on the MNIST dataset.

## Getting Started

1.  Navigate to one of the lab directories (`LAB01`, `LAB02`, or `LAB03`).
2.  Install the required packages: `pip install -r requirements.txt`
3.  Open and run the Jupyter notebook (`.ipynb` file) to follow the lab.
