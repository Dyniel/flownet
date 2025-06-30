# CFD GNN Project

This project provides a comprehensive suite for training and validating Graph Neural Network (GNN) models on Computational Fluid Dynamics (CFD) data. It supports noise injection, model training (FlowNet, RotFlowNet/Gao), various validation techniques (k-NN graph based, full mesh graph based, histogram JSD analysis), and extensive logging with Weights & Biases.

## Core Features

*   **Modular Library (`src/cfd_gnn/`)**: Reusable components for data processing, graph construction, GNN models, loss functions, metrics, training loops, and validation routines.
*   **Script-based Workflow (`scripts/`)**: Individual scripts for each stage of the pipeline:
    *   `1_prepare_noisy_data.py`: Injects MRI-like noise into CFD datasets.
    *   `2_train_model.py`: Main training script for GNN models.
    *   `3a_validate_knn.py`: Validates models using k-NN graphs.
    *   `3b_validate_full_mesh.py`: Validates models using full mesh (tetrahedral) graphs.
    *   `4_validate_histograms.py`: Performs standalone JSD histogram validation.
    *   `5_combined_validation.py`: Orchestrates inference and JSD validation.
*   **Configurable Pipeline**: Uses YAML configuration files (`config/default_config.yaml`) with CLI overrides for flexible experimentation.
*   **Weights & Biases Integration**: Comprehensive logging of metrics, configurations, and artifacts.
*   **Multiple Validation Strategies**: Supports both geometry-based graph construction (k-NN) and topology-based (full mesh from tetrahedra).

## Project Structure

```
.
├── config/
│   └── default_config.yaml       # Default configuration for all scripts
├── data/
│   └── .gitkeep                  # Placeholder for input CFD datasets (see below)
├── outputs/
│   └── .gitkeep                  # Default for generated outputs (models, predictions, logs) - gitignored
├── scripts/                      # Executable Python scripts for pipeline stages
│   ├── 1_prepare_noisy_data.py
│   ├── 2_train_model.py
│   ├── ...
├── src/
│   └── cfd_gnn/                  # Core library package
│       ├── __init__.py
│       ├── data_utils.py         # Data loading, noise, graph building, Dataset class
│       ├── losses.py             # Custom loss functions
│       ├── metrics.py            # Evaluation metrics (TKE, CosSim, JSD)
│       ├── models.py             # GNN model definitions (FlowNet, RotFlowNet)
│       ├── training.py           # Training and during-training validation loops
│       ├── utils.py              # General helper functions (config, seed, W&B, VTK I/O)
│       └── validation.py         # Standalone validation utilities (JSD pipeline)
├── tests/                        # (Optional) Unit and integration tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd cfd-gnn-project # Or your chosen directory name
    ```

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `torch`, `torch-scatter`, `torch-geometric` might require specific installation commands depending on your CUDA version. Refer to their official documentation if you encounter issues.*

4.  **Prepare Data**:
    *   Place your CFD datasets (typically series of `.vtk` files) into the `data/` directory.
    *   The expected structure for a dataset (e.g., "CFD_Ubend_other_val") is:
        ```
        data/
        └── CFD_Ubend_other_val/
            ├── sUbend_011/
            │   └── CFD/
            │       ├── Frame_00_data.vtk
            │       ├── Frame_01_data.vtk
            │       └── ...
            └── sUbend_012/
                └── CFD/
                    ├── Frame_00_data.vtk
                    └── ...
        ```
    *   Update paths in `config/default_config.yaml` (e.g., `train_root`, `val_root`) or provide them via CLI arguments to scripts if your data is elsewhere.

5.  **Weights & Biases (Optional)**:
    *   If you plan to use W&B logging, log in: `wandb login`
    *   You can specify your W&B project and entity in `config/default_config.yaml` or let the scripts use defaults.

## Configuration System

*   A `config/default_config.yaml` file provides default parameters for all aspects of the pipeline.
*   You can create custom YAML configuration files (e.g., `my_experiment_config.yaml`) and pass them to scripts using the `--config path/to/your_config.yaml` argument.
*   Command-line arguments provided directly to a script will override values from any loaded configuration file.

## Usage Examples

All scripts are run from the project root directory.

**1. Prepare Noisy Data:**
Creates a noisy version of a dataset.
```bash
python scripts/1_prepare_noisy_data.py \
    --source-dir data/CFD_Ubend_other_val \
    --output-dir outputs/noisy_data/CFD_Ubend_other_val_noisy \
    --p-min 0.05 \
    --p-max 0.15 \
    --overwrite
```
*   `--source-dir`: Directory of original CFD cases.
*   `--output-dir`: Where to save the noisy dataset.
*   `--p-min`, `--p-max`: Noise percentage range.
*   `--overwrite`: Overwrite output if it exists.
*   (See script help `python scripts/1_prepare_noisy_data.py -h` for more options)

**2. Train a Model:**
Trains FlowNet and/or Gao-RotFlowNet.
```bash
python scripts/2_train_model.py \
    --config config/my_training_setup.yaml \
    --run-name flownet_experiment_run01 \
    --models-to-train FlowNet \
    --epochs 150 \
    --no-wandb # Optionally disable W&B
```
*   `--config`: Path to your training config (can be omitted to use only defaults + CLI).
*   `--run-name`: Unique name for this training run; outputs will be in `outputs/<run_name>`.
*   `--models-to-train`: Specify one or more models (FlowNet, Gao/RotFlowNet).
*   (See script help for more options like LR, batch size, etc.)

**3. Validate Model with k-NN Graphs:**
Validates a trained model checkpoint using k-NN graph representation.
```bash
python scripts/3a_validate_knn.py \
    --model-checkpoint outputs/flownet_experiment_run01/models/flownet_best.pth \
    --model-name FlowNet \
    --val-data-dir data/CFD_Ubend_other_val \
    --output-dir outputs/flownet_experiment_run01/validation_knn \
    --k-neighbors 12 \
    --no-downsample
```
*   `--model-checkpoint`: Path to the `.pth` file of the trained model.
*   `--model-name`: Architecture name (FlowNet, Gao).
*   `--val-data-dir`: Directory of validation cases.
*   `--output-dir`: Where to save prediction VTKs and metrics summary.

**4. Validate Model with Full Mesh Graphs:**
Validates a trained model using graphs derived from mesh cell connectivity.
```bash
python scripts/3b_validate_full_mesh.py \
    --model-checkpoint outputs/flownet_experiment_run01/models/flownet_best.pth \
    --model-name FlowNet \
    --val-data-dir data/CFD_Ubend_other_val \
    --output-dir outputs/flownet_experiment_run01/validation_fullmesh
```
*   (Similar arguments to 3a, but without k-NN specific ones.)

**5. Perform Standalone JSD Histogram Validation:**
Compares two existing sets of VTK data (e.g., ground truth vs. model predictions).
```bash
python scripts/4_validate_histograms.py \
    --real-data-dir data/CFD_Ubend_other_val_noisy \
    --pred-data-dir outputs/flownet_experiment_run01/validation_knn/FlowNet \
    --output-dir outputs/flownet_experiment_run01/jsd_validation_knn \
    --velocity-key-real U_noisy \
    --velocity-key-pred velocity \
    --model-name-prefix FlowNet_KNN_vs_NoisyReal
```
*   `--real-data-dir`: Reference dataset.
*   `--pred-data-dir`: Predicted dataset (must have same case structure and frame count).
*   `--output-dir`: For JSD heatmap VTKs.
*   `--velocity-key-*`: VTK field names for velocity.

**6. Run Combined Validation (Inference + JSD):**
Orchestrates inference (like 3a or 3b) followed by JSD validation (like 4).
```bash
python scripts/5_combined_validation.py \
    --model-checkpoint outputs/flownet_experiment_run01/models/flownet_best.pth \
    --model-name FlowNet \
    --val-data-dir data/CFD_Ubend_other_val \
    --output-dir outputs/flownet_experiment_run01/combined_validation_knn_final \
    --graph-type knn \
    --k-neighbors 12
```
*   `--graph-type`: `knn` or `full_mesh` for the inference part.
*   This script will create subdirectories within its `--output-dir` for predictions and JSD results.

## Development Notes

*   **Device Management**: Scripts attempt to use CUDA if available and specified ("auto" or "cuda" in config/CLI). CPU is used as a fallback or if specified.
*   **VTK Keys**: Ensure the `velocity_key`, `pressure_key`, `noisy_velocity_key_suffix`, and `predicted_velocity_key` in your configuration match the fields in your VTK files and your desired output.
*   **Error Handling**: Scripts include basic error handling, but complex data issues might require debugging.
*   **Testing**: Consider adding unit tests to the `tests/` directory for core library functions.

## Future Enhancements (Ideas)

*   More sophisticated data augmentation techniques.
*   Support for additional GNN architectures.
*   Hyperparameter optimization scripts.
*   More detailed post-processing and visualization tools.
*   Integration with workflow management tools (e.g., Snakemake, Nextflow).
```
