# SHADES

This guide explains how to build the SHADES Docker image and run the scripts for statistics generation, model training, and empirical inference.

## Repository structure

Run all Docker commands from the root of the repository. The expected structure is:

```text
SHADES/
├── configs/
├── data/
├── models/
├── results/
├── src/
│   └── <scripts> 
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

The directories have the following purposes:

- `configs/` contains the YAML configuration files.
- `data/` contains the initial data and processed statistics.
- `models/` contains trained models.
- `results/` contains inference results.
- `src/` contains the Python source code.

Create the runtime directories if they do not already exist:

```bash
mkdir -p data models results
```

## Build the Docker image

From the repository root, run:

```bash
docker build -t shades:latest .
```

The image only needs to be rebuilt after changes to the Dockerfile, requirements, source code, or files copied into the image. Changes to mounted directories such as `data/`, `configs/`, `models/`, and `results/` do not require rebuilding the image.

## Confirm GPU access

Run the following command to confirm that PyTorch can access an NVIDIA GPU inside the container:

```bash
docker run --rm --gpus all \
  --entrypoint python \
  shades:latest \
  -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Directory mounts

The Docker commands below use bind mounts to make host directories available inside the container.

```text
Host directory    Container directory    Purpose
configs/          /app/configs           Configuration files
data/             /app/data              Input and processed data
models/           /app/models            Trained models
results/          /app/results           Inference results
```

The mounts are writable. Therefore, files created by a script inside these container directories are retained in the corresponding host directories.

# Statistics generation

We simulated CEU demographic data with SLiM 3. The simulation files use the `ms` format. The empirical scan uses files in the `vcf` format. Both formats can be parsed with the provided scripts to generate statistics.

Statistics generation requires the `data/` and `configs/` directories:

```bash
-v "$(pwd)/data:/app/data"
-v "$(pwd)/configs:/app/configs"
```

## `ms` files

Statistics must be generated separately for window sizes 11, 21, and 51.

### Window size 11

Generate images 1 and 3:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_data.py -c configs/parse_w11.yaml
```

Generate image 2:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/dist_from_freq.py -c configs/parse_w11.yaml
```

### Window size 21

Generate images 1 and 3:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_data.py -c configs/parse_w21.yaml
```

Generate image 2:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/dist_from_freq.py -c configs/parse_w21.yaml
```

### Window size 51

Generate images 1 and 3:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_data.py -c configs/parse_w51.yaml
```

Generate image 2:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/dist_from_freq.py -c configs/parse_w51.yaml
```

## `vcf` files

Statistics must also be generated separately for window sizes 11, 21, and 51 for empirical data.

### Window size 11

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_chromosome.py -c configs/parse_chromosome_w11.yaml
```

### Window size 21

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_chromosome.py -c configs/parse_chromosome_w21.yaml
```

### Window size 51

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_chromosome.py -c configs/parse_chromosome_w51.yaml
```

# Model training

The SHADES training process has two stages:

1. Train the base architecture, M1 or M2, separately with data for window sizes 11, 21, and 51.
2. Train the corresponding stacked SHADES model using the three trained base models while keeping their learned parameters frozen.

Training requires the `data/`, `configs/`, and `models/` directories:

```bash
-v "$(pwd)/data:/app/data"
-v "$(pwd)/configs:/app/configs"
-v "$(pwd)/models:/app/models"
```

## SHADES[M1]

### Train the M1 base models

Window size 11:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train.py -c configs/train_w11_m1.yaml
```

Window size 21:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train.py -c configs/train_w21_m1.yaml
```

Window size 51:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train.py -c configs/train_w51_m1.yaml
```

### Train SHADES[M1]

After all three M1 base models have been trained, train the stacked model:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train.py -c configs/train_sm1.yaml
```

## SHADES[M2]

### Train the M2 base models

Window size 11:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train_inception_resnet_v2.py -c configs/train_w11_m2.yaml
```

Window size 21:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train_inception_resnet_v2.py -c configs/train_w21_m2.yaml
```

Window size 51:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train_inception_resnet_v2.py -c configs/train_w51_m2.yaml
```

### Train SHADES[M2]

After all three M2 base models have been trained, train the stacked model:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  shades:latest \
  src/cnn_train_inc_resv2_stacking.py -c configs/train_sm2.yaml
```

# Empirical inference

The trained SHADES models can be used to scan the human genome and predict evolutionary classes.

Inference requires the `data/`, `configs/`, `models/`, and `results/` directories. The `models/` mount provides the trained models, and the `results/` mount preserves the inference output.

## SHADES[M1]

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  shades:latest \
  src/cnn_infer.py -c configs/infer_chromosome.yaml
```

## SHADES[M2]

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  shades:latest \
  src/cnn_infer_transfer_learning.py -c configs/infer_chromosome.yaml
```

## Running without GPU access

The image can be run without an exposed GPU by omitting `--gpus all`. The Python code selects the CPU when CUDA is unavailable.

For example:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  shades:latest \
  src/parse_data.py -c configs/parse_w11.yaml
```
