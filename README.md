# Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering


## 📝 Overview

This repository contains the official implementation of our paper:

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, and Konstantinos Tserpes   
**Conference**: 3rD Workshop on Advancements in Federated Learning @ ECML-PKDD (2025)  
**Paper**: [coming soon]  
**Citation**: [coming soon]

### 🔍 Abstract
> Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. 
We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side
dataset. This may correspond to, e.g., cases where the server possesses
trusted data prior to federation, or to the presence of a trusted client
that temporarily assumes the server role. Our approach requires only two
honest participants, i.e., the server and one client, to function effectively,
without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong
Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean,
Trimmed Mean, Median, Krum, and Multi-Krum under various attack
strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the
Flower framework.

## 🗂️ Project Structure

```bash
.
├── src/                # Core Flower code
│   ├── strategies/        # Flower server strategies (Loss-Based Clustering (ours), Mean, Trimmed Mean, Median, Krum, Multi Krum) 
├── configs/            # Configuration files for running and customizing experiments
├── data/               # Preprocessed partitioned data for FL clients
├── scripts/            # Data partitioning and simulation scripts
├── outputs/            # Timestamped outputs: logs, metrics, and best global model checkpoints per experiment
├── pyproject.toml      # Python project configuration and dependencies
└── README.md           # Project documentation and usage instructions
```

> **Note:** The `data` and `outputs` directories are created automatically upon executing the `partition_dataset` and `run_simulation` scripts, respectively.

## 🚀 Quick Start

### Prerequisites
Ensure you have [Poetry](https://python-poetry.org/docs/) installed and Python 3.12+ before proceeding.

### 1. Install Dependencies
Navigate to the root of the repository and install dependencies using Poetry:

```sh
poetry install
```

### 2. Preprocess and Distribute Data
Once dependencies are installed, preprocess the dataset and distribute it to the appropriate clients using the following command:

```sh
poetry run partition-dataset [OPTIONS]
```

#### Available Arguments
- `dataset_name` (Required): Name of the dataset.
- `--num_clients` (Optional): Number of federated learning (FL) clients.
- `--type` (Optional): Partitioning type, either `homogeneous` or `heterogeneous`. Default is `homogeneous`.
- `--alpha` (Optional): Alpha parameter for the Dirichlet distribution.

> **Note 1:** To reproduce the data partitioning used in the paper for the CIFAR-10, MNIST, and Fashion-MNIST datasets, execute the following commands:
> ```sh 
> poetry run partition-dataset CIFAR10 --num_clients=10 --type=homogeneous
> poetry run partition-dataset MNIST --num_clients=10 --type=homogeneous
> poetry run partition-dataset FMNIST --num_clients=10 --type=homogeneous
>```

> **Note 2:** At this stage, you may proceed with Steps 3 and 4 to configure your own simulation, or alternatively, execute the automated script used for the experiments presented in the paper by running
the following command: 
> ```sh 
> poetry run sh run_experiments.sh 
> ```
> The results of each experiment will be stored in timestamped directories within the `outputs` folder.


### 3. Set Configuration File
Set the configuration YAML file as an environment variable:

```sh
export config_file_name=config
```
The configs directory contains predefined YAML configuration files designed for simulating various attacks, such as `config_data_attack` for data specific attacks 
and `config_model_attack` for model specific attacks.
To apply a specific configuration, simply update the corresponding environment variable with the desired YAML file name.

### 4. Run the Simulation
Execute the simulation script using Poetry:

```sh
poetry run simulation
```

