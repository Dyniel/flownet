import wandb
import yaml
import argparse
from pathlib import Path
import sys

# Ensure the project root is in the Python path for utils if needed
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# from src.cfd_gnn.utils import load_config # If needed for advanced setup

def main():
    parser = argparse.ArgumentParser(description="Launch a Weights & Biases sweep.")
    parser.add_argument(
        "--sweep-config", type=str, default="sweep_config.yaml",
        help="Path to the W&B sweep configuration YAML file."
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="W&B project name. Overrides project in sweep_config or default."
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="W&B entity (username or team). Overrides entity in sweep_config."
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of runs for the sweep agent to execute. Default is to run indefinitely or until sweep stops."
    )

    args = parser.parse_args()

    # Load sweep configuration from YAML
    # The sweep_config.yaml is expected to be in the root directory, relative to where this script is run from.
    # If this script is run from the root, 'sweep_config.yaml' is correct.
    # If this script is run from 'scripts/', then the path should be '../sweep_config.yaml'.
    # For simplicity, assuming it's run from the project root.
    sweep_config_path = Path(args.sweep_config)
    if not sweep_config_path.is_absolute():
        # If run from scripts/ a common pattern is that CWD is project root
        # However, to be robust, let's assume it's relative to project_root if not absolute
        # This means sweep_config.yaml should be at project_root/sweep_config.yaml
         pass # Keep it as is, user expected to provide correct path or have it in root.

    try:
        with open(sweep_config_path, 'r') as f:
            sweep_yaml_content = f.read()
            sweep_configuration = yaml.safe_load(sweep_yaml_content)
            print(f"Loaded sweep configuration from: {str(sweep_config_path)}")
    except FileNotFoundError:
        print(f"Error: Sweep configuration file not found at {str(sweep_config_path)}")
        print(f"Please ensure '{args.sweep_config}' exists in the correct location (e.g., project root).")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing sweep configuration file: {e}")
        sys.exit(1)

    # Determine W&B project and entity
    project_name = args.project or sweep_configuration.get("project")
    if not project_name:
        try:
            default_cfg_path = project_root / "config" / "default_config.yaml"
            with open(default_cfg_path, 'r') as f_cfg:
                default_global_cfg = yaml.safe_load(f_cfg)
            project_name = default_global_cfg.get("wandb_project", "CFD_GNN_Sweep_Default")
            print(f"Inferred W&B project from default_config.yaml: {project_name}")
        except Exception:
            project_name = "CFD_GNN_Sweep_Default"
            print(f"Warning: W&B project name not specified. Using default: {project_name}")


    entity_name = args.entity or sweep_configuration.get("entity")

    print(f"Sweep will run in project: {project_name}" + (f", entity: {entity_name}" if entity_name else ""))

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=project_name,
        entity=entity_name
    )
    print(f"Sweep initialized. Sweep ID: {sweep_id}")
    agent_command_parts = ["wandb", "agent"]
    if entity_name:
        agent_command_parts.append(f"{entity_name}/{project_name}/{sweep_id}")
    else:
        agent_command_parts.append(f"{project_name}/{sweep_id}") # Default entity

    print(f"Run `{' '.join(agent_command_parts)}` in your terminal to start an agent manually.")
    print("Or, this script will now attempt to start a local agent.")

    try:
        print(f"Starting W&B agent for sweep {sweep_id}...")
        wandb.agent(sweep_id, project=project_name, entity=entity_name, count=args.count)
        print("W&B agent finished.")
    except Exception as e:
        print(f"An error occurred while running the W&B agent: {e}")
        print("You might need to run the agent manually in your terminal using the command printed above.")

if __name__ == "__main__":
    main()
