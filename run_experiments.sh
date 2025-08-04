#!/bin/bash

DATASETS="CIFAR10 FMNIST MNIST"
STRATEGIES="Loss-Based-Clustering Krum Multi-Krum Median Trimmed-Mean Mean"
MALICIOUS_USERS="5"

for m in $MALICIOUS_USERS
do
  for d in $DATASETS
  do
    for s in $STRATEGIES
    do

      echo "Handling no attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="configs/config_no_attack.yaml"
      yq -i -y ".server.strategy = \"$s\" | .model.name = \"$d\"" "$YAML_FILE"
      export config_file_name="config_no_attack"
      poetry run simulation

      echo "Handling Label-FLipping attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="configs/config_data_attack.yaml"
      yq -i -y ".server.strategy = \"$s\" | .attack.num_malicious_clients = $m | .model.name = \"$d\"" "$YAML_FILE"
      export config_file_name="config_data_attack"
      poetry run simulation

      echo "Handling Gaussian Noise attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="configs/config_model_attack.yaml"
      yq -i -y ".server.strategy = \"$s\" | .attack.num_malicious_clients = $m | .model.name = \"$d\" | .attack.type = \"Gaussian Noise\"" "$YAML_FILE"
      export config_file_name="config_model_attack"
      poetry run simulation

      echo "Handling Sign Flipping attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="configs/config_model_attack.yaml"
      yq -i -y ".server.strategy = \"$s\" | .attack.num_malicious_clients = $m | .model.name = \"$d\" | .attack.type = \"Sign Flip\"" "$YAML_FILE"
      export config_file_name="config_model_attack"
      poetry run simulation

    done
  done
done
