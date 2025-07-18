import yaml
import subprocess
import os
import copy # For deep copying configurations

# Define the base configuration file path
BASE_CONFIG_PATH = "config/test_config.yaml"
# Define the path for temporary configuration files
TEMP_CONFIG_DIR = "outputs/experiment_configs"
TEMP_CONFIG_NAME = "temp_experiment_config.yaml"
# Define the training script path
TRAIN_SCRIPT_PATH = "scripts/2_train_model.py"
# Default number of epochs for experiments
DEFAULT_EPOCHS = 75 # Increased default epochs

# --- Define your experiments here ---
# Each dictionary in this list represents one experiment.
# Parameters specified here will override those in the BASE_CONFIG_PATH.
# Add a 'run_name_suffix' to easily identify experiments in W&B.
# All experiments will use FlowNet by default as per test_config.yaml
# We can specify models_to_train: ["FlowNet"] or ["Gao"] if needed.

experiments = [
    # Baseline (uses defaults from test_config.yaml, with DEFAULT_EPOCHS)
    {
        "run_name_suffix": "flownet_baseline",
        "models_to_train": ["FlowNet"],
    },

    # Learning Rate Variations (for FlowNet)
    {
        "run_name_suffix": "flownet_lr_1e-5",
        "lr": 1.0e-5, # Overrides top-level lr
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_lr_5e-5",
        "lr": 5.0e-5,
        "models_to_train": ["FlowNet"],
    },
    # lr: 2.0e-4 is default in test_config
    {
        "run_name_suffix": "flownet_lr_5e-4",
        "lr": 5.0e-4,
        "models_to_train": ["FlowNet"],
    },

    # Hidden Dimension Variations (for FlowNet)
    {
        "run_name_suffix": "flownet_hdim_64",
        "model_config": {"h_dim": 64}, # test_config default is 128
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_hdim_256",
        "model_config": {"h_dim": 256},
        "models_to_train": ["FlowNet"],
    },

    # Number of Layers Variations (for FlowNet)
    {
        "run_name_suffix": "flownet_layers_3",
        "model_config": {"layers": 3}, # test_config default is 5
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_layers_6",
        "model_config": {"layers": 6},
        "models_to_train": ["FlowNet"],
    },

    # Loss Weights Variations (for FlowNet)
    # Baseline loss weights from test_config.yaml: {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
    {
        "run_name_suffix": "flownet_loss_heavy_div",
        "loss_config": {"weights": {"supervised": 0.8, "divergence": 0.2, "histogram": 0.0}},
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_loss_balanced",
        "loss_config": {"weights": {"supervised": 0.7, "divergence": 0.15, "histogram": 0.15}},
        "models_to_train": ["FlowNet"],
    },

    # Graph Downsampling Variations (for FlowNet)
    # graph_config.down_n default is 20000 in test_config.yaml
    {
        "run_name_suffix": "flownet_downN_10k",
        "graph_config": {"down_n": 10000},
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_downN_none_b2", # No downsampling, smaller batch
        "graph_config": {"down_n": None}, # null in YAML often becomes None in Python
        "batch_size": 2, # Overrides top-level batch_size
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_downN_none_b1", # No downsampling, smallest batch
        "graph_config": {"down_n": None},
        "batch_size": 1,
        "models_to_train": ["FlowNet"],
    },

    # Batch Size Variation (with default down_n=20000) (for FlowNet)
    # batch_size default is 4 in test_config.yaml
    {
        "run_name_suffix": "flownet_batch_2",
        "batch_size": 2,
        "models_to_train": ["FlowNet"],
    },

    # Example for Gao model (can be expanded similarly)
    {
        "run_name_suffix": "gao_baseline",
        "models_to_train": ["Gao"],
        # Uses defaults from test_config for Gao, if specific Gao configs are present
        # or general model_config if not.
    },
    {
        "run_name_suffix": "gao_lr_1e-5",
        "lr": 1.0e-5,
        "models_to_train": ["Gao"],
    },
    {
        "run_name_suffix": "gao_downN_none_b1",
        "graph_config": {"down_n": None},
        "batch_size": 1,
        "models_to_train": ["Gao"],
    },

    # --- Regularization Experiments (for FlowNet) ---
    # Assuming regularization_type and regularization_lambda are top-level config keys
    # (as in default_config.yaml)
    {
        "run_name_suffix": "flownet_L1_1e-5",
        "regularization_type": "L1",
        "regularization_lambda": 1e-5,
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_L1_1e-4",
        "regularization_type": "L1",
        "regularization_lambda": 1e-4,
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_L1_1e-3",
        "regularization_type": "L1",
        "regularization_lambda": 1e-3,
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_L2_1e-5",
        "regularization_type": "L2",
        "regularization_lambda": 1e-5,
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_L2_1e-4",
        "regularization_type": "L2",
        "regularization_lambda": 1e-4,
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_L2_1e-3",
        "regularization_type": "L2",
        "regularization_lambda": 1e-3,
        "models_to_train": ["FlowNet"],
    },

    # --- Graph Parameter 'k' (for k-NN) Variations (for FlowNet) ---
    # Default k is likely 12 (from default_config, assuming test_config follows)
    {
        "run_name_suffix": "flownet_k_6",
        "graph_config": {"k": 6},
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_k_18",
        "graph_config": {"k": 18},
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_k_24",
        "graph_config": {"k": 24},
        "models_to_train": ["FlowNet"],
    },

    # --- GNN Internal MLP Layer Variations (for FlowNet) ---
    # Defaults are typically 2 for encoder_mlp_layers, decoder_mlp_layers, gnn_step_mlp_layers
    # Let's vary gnn_step_mlp_layers, assuming it's a top-level config picked up by model
    {
        "run_name_suffix": "flownet_gnnstepmlp_1",
        "gnn_step_mlp_layers": 1, # Top-level override
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_gnnstepmlp_3",
        "gnn_step_mlp_layers": 3, # Top-level override
        "models_to_train": ["FlowNet"],
    },
    {
        "run_name_suffix": "flownet_encmlp_1_decmlp_1", # Example varying encoder/decoder
        "encoder_mlp_layers": 1, # Top-level override
        "decoder_mlp_layers": 1, # Top-level override
        "models_to_train": ["FlowNet"],
    }
]

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def main():
    # Create directory for temporary configs if it doesn't exist
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, TEMP_CONFIG_NAME)

    # Load the base configuration
    try:
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Base configuration file not found at {BASE_CONFIG_PATH}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing base configuration file: {e}")
        return

    print(f"Loaded base configuration from {BASE_CONFIG_PATH}")
    print(f"Found {len(experiments)} experiments to run.")
    print("---")

    for i, exp_overrides in enumerate(experiments):
        print(f"Running experiment {i+1}/{len(experiments)}: {exp_overrides.get('run_name_suffix', 'unnamed_experiment')}")

        # Create a deep copy of the base config for this experiment
        current_config = copy.deepcopy(base_config)

        # Apply overrides
        # We need to handle which keys are being overridden from the exp_overrides
        config_to_update_with = {k: v for k, v in exp_overrides.items() if k != 'run_name_suffix'}
        current_config = deep_update(current_config, config_to_update_with)

        # Ensure epochs are set, either from experiment or default
        if "training_settings" not in current_config:
            current_config["training_settings"] = {}
        current_config["training_settings"]["epochs"] = exp_overrides.get("epochs", DEFAULT_EPOCHS)

        # Construct run name for W&B
        # The run name in W&B will be controlled by --run-name argument to 2_train_model.py
        # We can use the project name from the config and append the suffix.
        project_name = current_config.get("wandb_settings", {}).get("project", "CFD_GNN_Refactored")
        run_name_suffix = exp_overrides.get('run_name_suffix', f'exp_{i+1}')
        # The actual run name passed to the script.
        # `2_train_model.py` uses its --run-name argument for the output directory and W&B run name.
        # Let's form a descriptive name for the output directory.
        # Example: outputs/MyProject_baseline, outputs/MyProject_low_lr
        # The wandb run name will be exactly this.
        # The `run_name` in test_config.yaml is for the output subfolder, this will be overridden by the CLI arg.

        # The `run_name` parameter in the YAML config usually defines the output sub-directory.
        # The `--run-name` CLI argument to `2_train_model.py` overrides W&B run name AND this output folder.
        # So we just need to set the CLI argument.
        unique_run_name = f"{project_name}_{run_name_suffix}"


        # Save the modified configuration to the temporary file
        try:
            with open(temp_config_path, 'w') as f:
                yaml.dump(current_config, f, sort_keys=False)
            print(f"  Saved temporary config to {temp_config_path}")
        except IOError as e:
            print(f"  Error saving temporary configuration file: {e}")
            continue # Skip to next experiment

        # Construct the command
        command = [
            "python",
            TRAIN_SCRIPT_PATH,
            "--config",
            temp_config_path,
            "--run-name",
            unique_run_name, # This will be the W&B run name and output folder name
            "--epochs", # Ensure epochs from config are overridden by CLI if needed, or rely on config.
            str(current_config["training_settings"]["epochs"]) # Use epochs from current_config
        ]

        # Add models_to_train if specified in experiment, otherwise it defaults to what's in test_config
        if "models_to_train" in exp_overrides: # Assuming models_to_train is a list
             command.extend(["--models-to-train"] + exp_overrides["models_to_train"])
        elif "model_name" in exp_overrides: # Handle single model_name to models_to_train
             command.extend(["--models-to-train", exp_overrides["model_name"]])


        print(f"  Executing command: {' '.join(command)}")

        # Run the training script
        try:
            process = subprocess.run(command, check=True, text=True, capture_output=False) # Set capture_output to False
            print(f"  Experiment {run_name_suffix} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  Error running experiment {run_name_suffix}:")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"  Error: Could not find training script at {TRAIN_SCRIPT_PATH} or python interpreter.")

        print("---")

    print("All experiments finished.")
    # Optionally, clean up the temporary config file
    # if os.path.exists(temp_config_path):
    #     os.remove(temp_config_path)

if __name__ == "__main__":
    main()
