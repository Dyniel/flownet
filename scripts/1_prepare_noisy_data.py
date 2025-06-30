#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_prepare_noisy_data.py
-----------------------
Script to create a "noisy" version of a CFD dataset.
It iterates through specified cases, reads VTK files, injects MRI-like noise
to point positions and velocity fields, and saves the new VTK files to an
output directory, preserving the original structure.

Usage:
    python scripts/1_prepare_noisy_data.py --config path/to/your_config.yaml [options]

Example:
    python scripts/1_prepare_noisy_data.py \
        --source-dir data/CFD_Ubend_other \
        --output-dir outputs/noisy_data/CFD_Ubend_other_noisy \
        --p-min 0.05 --p-max 0.15 \
        --overwrite

If --config is provided, its values are used as defaults, overridden by CLI args.
"""

import argparse
from pathlib import Path
import sys

# Ensure the src directory is in the Python path
# This allows direct execution of scripts from the project root, e.g., python scripts/script.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import create_noisy_dataset_tree
from src.cfd_gnn.utils import load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Prepare a noisy version of a CFD dataset.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--source-dir", type=str, help="Source directory containing original CFD cases."
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save the noisy dataset."
    )
    parser.add_argument(
        "--p-min", type=float, help="Minimum noise percentage (e.g., 0.05 for 5%)."
    )
    parser.add_argument(
        "--p-max", type=float, help="Maximum noise percentage (e.g., 0.15 for 15%)."
    )
    parser.add_argument(
        "--velocity-key", type=str, help="Key for the velocity field in input VTKs (e.g., 'U')."
    )
    parser.add_argument(
        "--no-position-noise", action="store_true", help="If set, do not add noise to point positions."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="If set, overwrite output directory if it exists."
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for noise generation reproducibility."
    )

    args = parser.parse_args()

    # Load default config and update with CLI args
    # The config file might define 'train_root' or 'val_root' which could serve as source_dir
    # and 'noisy_train_root' or 'noisy_val_root' for output_dir
    default_cfg_path = project_root / "config" / "default_config.yaml"
    cfg = load_config(args.config or default_cfg_path) # Load user config or default

    # Prioritize CLI arguments over config file values
    source_dir = Path(args.source_dir or cfg.get("train_root", cfg.get("val_root"))) # Example fallback
    output_dir = Path(args.output_dir or cfg.get("noisy_train_root", cfg.get("noisy_val_root"))) # Example fallback

    p_min = args.p_min if args.p_min is not None else cfg.get("p_min", 0.05)
    p_max = args.p_max if args.p_max is not None else cfg.get("p_max", 0.15)
    velocity_key = args.velocity_key or cfg.get("velocity_key", "U")
    position_noise = not args.no_position_noise # Default is True (add position noise)
    overwrite_flag = args.overwrite
    seed_value = args.seed if args.seed is not None else cfg.get("seed")

    if not source_dir or not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' not found or not specified.")
        parser.print_help()
        sys.exit(1)

    if not output_dir:
        print(f"Error: Output directory not specified.")
        parser.print_help()
        sys.exit(1)


    if seed_value is not None:
        set_seed(seed_value)
        print(f"Using random seed: {seed_value}")

    print(f"Starting noisy data preparation:")
    print(f"  Source dataset: {source_dir}")
    print(f"  Output (noisy) dataset: {output_dir}")
    print(f"  Noise range: {p_min*100:.1f}% to {p_max*100:.1f}%")
    print(f"  Velocity field key: '{velocity_key}'")
    print(f"  Add noise to positions: {position_noise}")
    print(f"  Overwrite if exists: {overwrite_flag}")

    create_noisy_dataset_tree(
        source_root=source_dir,
        destination_root=output_dir,
        p_min=p_min,
        p_max=p_max,
        velocity_key=velocity_key,
        position_noise=position_noise,
        overwrite=overwrite_flag
    )

    print("Noisy data preparation script finished.")

if __name__ == "__main__":
    main()
