#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_train_model.py
----------------
Main script for training GNN models (FlowNet and/or Gao-RotFlowNet) on CFD data.

Pipeline:
 1. Loads configuration (default + user YAML + CLI).
 2. Initializes W&B (if enabled) and sets random seed.
 3. Prepares datasets (training on noisy data, validation on noisy or clean data).
 4. Initializes model(s), optimizer(s), and scheduler(s).
 5. Runs the training loop:
    - Trains for one epoch using `train_single_epoch`.
    - Validates periodically using `validate_on_pairs`.
    - Logs metrics to W&B and CSV.
    - Implements early stopping.
    - Saves best model checkpoints.
 6. Optionally, performs histogram JSD validation after training.

Usage:
    python scripts/2_train_model.py --config path/to/config.yaml --run-name my_run [options]

Example:
    python scripts/2_train_model.py --run-name flownet_test --models FlowNet --epochs 50
"""

import argparse
import csv
import time
from pathlib import Path
import sys
import shutil
import wandb
import pandas as pd # For saving probe data
import numpy as np # For np.nan

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader


# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import PairedFrameDataset, make_frame_pairs, vtk_to_knn_graph, vtk_to_fullmesh_graph
from src.cfd_gnn.models import FlowNet, RotFlowNet, FlowNetGATv2 # Add FlowNetGATv2
from src.cfd_gnn.training import train_single_epoch, validate_on_pairs
from src.cfd_gnn.validation import histogram_jsd_validation # For post-training validation
from src.cfd_gnn.utils import (
    load_config, set_seed, get_run_name, initialize_wandb, get_device,
    sort_frames_by_number # For physical metrics on a sample if needed
)
from src.cfd_gnn.metrics import turbulent_kinetic_energy, cosine_similarity_metric


def main():
    parser = argparse.ArgumentParser(description="Train GNN models for CFD prediction.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name for this run (used for output directories and W&B)."
    )
    parser.add_argument(
        "--models-to-train", type=str, nargs="+", default=["FlowNet"],
        choices=["FlowNet", "Gao", "RotFlowNet", "FlowNetGATv2"],
        help="Which model architectures to train (e.g., FlowNet FlowNetGATv2)."
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of training epochs from config."
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size from config."
    )
    parser.add_argument(
        "--lr", type=float, help="Override learning rate from config."
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "auto"], help="Override device from config."
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--output-base-dir", type=str, help="Override base directory for outputs."
    )
    parser.add_argument(
        "--skip-hist-val-after-train", action="store_true", help="Skip histogram JSD validation after training."
    )
    parser.add_argument(
        "--data-source", type=str, default="noisy", choices=["noisy", "clean"],
        help="Specify whether to use 'noisy' or 'clean' dataset for training and primary validation. Default: noisy."
    )

    args = parser.parse_args()

    # --- 1. Configuration Loading ---
    default_cfg_path = project_root / "config" / "default_config.yaml"
    cfg = load_config(args.config or default_cfg_path)

    # Override config with CLI arguments if provided
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.lr is not None: cfg["lr"] = args.lr
    if args.device is not None: cfg["device"] = args.device
    if args.output_base_dir is not None: cfg["output_base_dir"] = args.output_base_dir

    run_name = get_run_name(args.run_name)
    cfg["run_name"] = run_name # Store in config for W&B and other uses

    # Determine device
    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    # Extract Reynolds number and check if required
    physics_cfg = cfg.get("physics", {})
    reynolds_number = physics_cfg.get("reynolds_number")

    # Loss configuration (weights and dynamic balancing)
    loss_cfg_main = cfg.get("loss_config", {})
    loss_weights_main_cfg = loss_cfg_main.get("weights", {})
    dynamic_balancing_cfg_main = loss_cfg_main.get("dynamic_loss_balancing", {"enabled": False}) # Get DLB config

    print(f"DEBUG: Reynolds number from config: {reynolds_number}")
    print(f"DEBUG: Loss weights from config: {loss_weights_main_cfg}")
    print(f"DEBUG: Dynamic Loss Balancing config: {dynamic_balancing_cfg_main}")

    if loss_weights_main_cfg.get("navier_stokes", 0.0) > 0.0 and reynolds_number is None:
        raise ValueError(
            "Reynolds number (`physics.reynolds_number`) must be provided in config "
            "if 'navier_stokes' loss weight is > 0."
        )

    # --- 2. Initialization ---
    if args.no_wandb or cfg.get("no_wandb", False):
        wandb_run = None
        print("Weights & Biases logging disabled.")
    else:
        wandb_run = initialize_wandb(
            config=cfg, run_name=run_name,
            project_name=cfg.get("wandb_project"),
            entity=cfg.get("wandb_entity"),
            output_dir=project_root / cfg.get("output_base_dir", "outputs") / run_name
        )

    set_seed(cfg["seed"], device_specific=(device.type == "cuda"))
    print(f"Random seed set to: {cfg['seed']}")

    # Output directory setup
    run_output_dir = project_root / cfg.get("output_base_dir", "outputs") / run_name
    models_output_dir = run_output_dir / "models"
    hist_val_output_dir = run_output_dir / "histograms_after_training"
    models_output_dir.mkdir(parents=True, exist_ok=True)
    # hist_val_output_dir will be created by histogram_validation if run

    print(f"Run name: {run_name}")
    print(f"Outputs will be saved to: {run_output_dir}")

    # --- 3. Data Preparation ---
    if args.data_source == "clean":
        train_data_path_to_use = Path(cfg["train_root"])
        train_use_noisy_flag = False
        print("Using CLEAN dataset for training.")
    else: # Default is "noisy"
        train_data_path_to_use = Path(cfg["noisy_train_root"])
        train_use_noisy_flag = True
        print("Using NOISY dataset for training.")

    # Validation data for during-training checks - also respects --data-source by default
    val_during_train_cfg = cfg.get("validation_during_training", {})
    # If --data-source is clean, validation should also be clean by default.
    # Allow override from val_during_train_cfg if specific advanced control is needed,
    # but primary driver is args.data_source.
    default_val_use_noisy = train_use_noisy_flag # Match training source by default
    val_use_noisy_for_run = val_during_train_cfg.get("use_noisy_data", default_val_use_noisy)

    if val_use_noisy_for_run:
        val_data_path_to_use = Path(cfg["noisy_val_root"])
        print("Using NOISY dataset for validation during training.")
    else:
        val_data_path_to_use = Path(cfg["val_root"])
        print("Using CLEAN dataset for validation during training.")


    print(f"Loading training data from: {train_data_path_to_use}")
    print(f"Loading validation data (during training) from: {val_data_path_to_use}")

    train_pairs = make_frame_pairs(train_data_path_to_use)
    val_pairs_during_train = make_frame_pairs(val_data_path_to_use)

    if not train_pairs:
        print(f"Error: No training frame pairs found in {train_data_path_to_use}. Exiting.")
        sys.exit(1)
    if not val_pairs_during_train and val_during_train_cfg.get("enabled", True):
        print(f"Warning: No validation frame pairs found in {val_data_path_to_use} for during-training validation.")
        # Allow continuation if validation is optional or handled differently

    graph_build_config_train = {
        **cfg.get("graph_config",{}),
        "velocity_key": cfg["velocity_key"],
        "pressure_key": cfg.get("pressure_key", "p"), # Added pressure_key for kNN if used
        "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]
    }
    graph_type_train = cfg.get("default_graph_type", "knn")
    print(f"DEBUG: Training graph build config: {graph_build_config_train}")
    print(f"DEBUG: Training graph type: {graph_type_train}")

    train_dataset = PairedFrameDataset(
        train_pairs, graph_build_config_train, graph_type=graph_type_train,
        use_noisy_data=train_use_noisy_flag # Controlled by --data-source
        # device=device # Removed: Graphs are created on CPU by PairedFrameDataset
    )
    train_loader = PyGDataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers",0))

    # --- CSV Logger Setup ---
    csv_log_path = run_output_dir / "training_metrics.csv"
    csv_file = open(csv_log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "epoch", "model_name",
        "train_loss_total", "train_loss_sup", "train_loss_div",
        "train_loss_ns_mom", "train_loss_lbc",
        "train_loss_hist", "train_loss_reg",
        "train_alpha_dlb", "train_beta_dlb",
        "val_mse", "val_rmse_mag", "val_mse_div", "val_cosine_sim",
        "val_mse_x", "val_mse_y", "val_mse_z",
        "val_perc_points_within_10_rel_err",
        "val_nrmse_vel", "val_nrmse_p", # Added NRMSE metrics
        "val_avg_max_true_vel_mag", "val_avg_max_pred_vel_mag",
        "lr", "sample_rel_tke_err", "sample_cosine_sim"
    ]
    csv_writer.writerow(csv_header)
    print(f"Logging training metrics to: {csv_log_path}")

    # --- Models to Train ---
    models_registry = {
        "FlowNet": FlowNet,
        "Gao": RotFlowNet, # Assuming Gao is RotFlowNet
        "RotFlowNet": RotFlowNet,
        "FlowNetGATv2": FlowNetGATv2
    }
    models_to_train_names = [name for name in args.models_to_train if name in models_registry]
    if not models_to_train_names:
        print("Error: No valid models specified for training. Check --models-to-train argument.")
        sys.exit(1)

    print(f"Models to train: {models_to_train_names}")

    # --- 4. Training Loop for each model ---
    for model_name in models_to_train_names:
        print(f"\n===== Training Model: {model_name} =====")

        # Initialize model, optimizer, scheduler
        model_arch_cfg = {**cfg, **cfg.get(f"{model_name}_config", {})} # Allow model-specific overrides

        # --- Logging model and graph parameters ---
        print(f"DEBUG: Initializing model {model_name} with the following effective parameters:")
        print(f"  h_dim (hidden_dim): {model_arch_cfg.get('h_dim', 128)}") # Default from FlowNet if not in cfg
        print(f"  layers (num_gnn_layers): {model_arch_cfg.get('layers', 5)}")
        print(f"  encoder_mlp_layers: {model_arch_cfg.get('encoder_mlp_layers', 2)}")
        print(f"  decoder_mlp_layers: {model_arch_cfg.get('decoder_mlp_layers', 2)}")
        print(f"  gnn_step_mlp_layers: {model_arch_cfg.get('gnn_step_mlp_layers', 2)}")
        print(f"  node_in_features: {model_arch_cfg.get('node_in_features', 3)}") # Default from FlowNet
        print(f"  edge_in_features: {model_arch_cfg.get('edge_in_features', 3)}") # Default from FlowNet
        print(f"  checkpoint_edge_encoder_internals: {model_arch_cfg.get('checkpoint_edge_encoder_internals', False)}") # Default from FlowNet

        graph_cfg_params = cfg.get("graph_config", {})
        print(f"DEBUG: Graph construction parameters (from 'graph_config'):")
        print(f"  k (for kNN): {graph_cfg_params.get('k', 'Not set in graph_config, default in data_utils might apply')}")
        print(f"  down_n (downsampling): {graph_cfg_params.get('down_n', 'Not set, no downsampling or default in data_utils')}")
        # --- End logging ---

        model = models_registry[model_name](model_arch_cfg).to(device)
        optimizer = Adam(model.parameters(), lr=cfg["lr"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg["patience"] // 2) # Adjusted patience for scheduler

        best_val_metric = float('inf')
        epochs_no_improve = 0

        # Per-model state for early stopping and best model saving
        best_model_path = models_output_dir / f"{model_name.lower()}_best.pth"
        all_probes_data_for_model = [] # Initialize list to collect probe data across epochs for this model

        for epoch in range(1, cfg["epochs"] + 1):
            epoch_start_time = time.time()

            train_metrics = train_single_epoch(
                model, train_loader, optimizer,
                loss_weights=loss_weights_main_cfg,
                reynolds_number=reynolds_number,
                histogram_bins=cfg.get("loss_config",{}).get("histogram_bins", cfg["nbins"]),
                device=device,
                clip_grad_norm_value=cfg.get("clip_grad_norm"),
                regularization_type=cfg.get("regularization_type", "None"),
                regularization_lambda=cfg.get("regularization_lambda", 0.0),
                dynamic_balancing_cfg=dynamic_balancing_cfg_main # Pass DLB config
            )
            print(f"DEBUG: Epoch {epoch} Train Metrics: {train_metrics}")

            current_lr = optimizer.param_groups[0]['lr']
            val_metrics = {} # Initialize val_metrics here to ensure it always exists

            log_dict_epoch = {
                f"{model_name}/train_loss_total": train_metrics.get("total", np.nan),
                f"{model_name}/train_loss_sup": train_metrics.get("supervised", np.nan),
                f"{model_name}/train_loss_div": train_metrics.get("divergence", np.nan),
                f"{model_name}/train_loss_ns_mom": train_metrics.get("navier_stokes_momentum", np.nan),
                f"{model_name}/train_loss_lbc": train_metrics.get("lbc", np.nan),
                f"{model_name}/train_loss_hist": train_metrics.get("histogram", np.nan),
                f"{model_name}/train_loss_reg": train_metrics.get("regularization", np.nan),
                f"{model_name}/train_alpha_dlb": train_metrics.get("alpha_final_batch", np.nan),
                f"{model_name}/train_beta_dlb": train_metrics.get("beta_final_batch", np.nan),
                f"{model_name}/learning_rate": current_lr,
                "epoch": epoch
            }
            # Add NRMSE to W&B log if available in val_metrics
            if "val_nrmse_vel" in val_metrics:
                log_dict_epoch[f"{model_name}/val_nrmse_vel"] = val_metrics["val_nrmse_vel"]
            if "val_nrmse_p" in val_metrics:
                log_dict_epoch[f"{model_name}/val_nrmse_p"] = val_metrics["val_nrmse_p"]

            # Perform validation (during training)
            val_metrics = {}
            if val_during_train_cfg.get("enabled", True) and val_pairs_during_train and \
               (epoch % val_during_train_cfg.get("frequency", 1) == 0 or epoch == cfg["epochs"]):

                val_graph_cfg = {**cfg.get("graph_config",{}), **val_during_train_cfg.get("val_graph_config",{})}
                val_graph_cfg.update({"velocity_key": cfg["velocity_key"], "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]})
                val_graph_type = val_during_train_cfg.get("val_graph_type", graph_type_train) # graph_type is used by validate_on_pairs if not in val_graph_cfg

                save_fields_vtk_flag = val_during_train_cfg.get("save_validation_fields_vtk", False)
                log_field_image_idx = val_during_train_cfg.get("log_field_image_sample_idx", 0)

                # Pass global_cfg to validate_on_pairs for access to all configs including probes
                # It now returns (val_metrics, epoch_probe_data_for_csv)
                # W&B Table logging for probes is handled inside validate_on_pairs
                val_metrics, epoch_probe_data_for_csv = validate_on_pairs(
                    model=model,
                    val_frame_pairs=val_pairs_during_train,
                    global_cfg=cfg, # Pass the main config dict
                    use_noisy_data_for_val=val_use_noisy_for_run,
                    device=device,
                    epoch_num=epoch,
                    output_base_dir=run_output_dir,
                    save_fields_vtk=save_fields_vtk_flag,
                    wandb_run=wandb_run,
                    log_field_image_sample_idx=log_field_image_idx,
                    model_name=model_name
                )
                if epoch_probe_data_for_csv: # If any probe data was collected this epoch for CSV
                    all_probes_data_for_model.extend(epoch_probe_data_for_csv)

                log_dict_epoch.update({f"{model_name}/{k}": v for k,v in val_metrics.items()})
                # Removed merging of epoch_probe_data_for_wandb as it's handled by Table logging internally

                # Check for improvement (using val_mse as primary metric)
                if val_metrics.get("val_mse", float('inf')) < best_val_metric:
                    best_val_metric = val_metrics["val_mse"]
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Epoch {epoch}: New best model for {model_name} saved to {best_model_path} (Val MSE: {best_val_metric:.4e})")
                else:
                    epochs_no_improve += 1

                scheduler.step(val_metrics.get("val_mse", float('inf')))


            # Quick physical metrics on a single validation sample (from original script)
            # This can be made more robust or optional
            sample_rel_tke_err, sample_cos_sim = -1.0, -1.0 # Default if not computed
            if val_pairs_during_train: # Ensure there's data to sample from
                try:
                    sample_path_t0, sample_path_t1 = val_pairs_during_train[0] # Use first pair
                    # Use same graph config as validation for consistency
                    sample_graph_cfg = {**cfg.get("graph_config",{}), **val_during_train_cfg.get("val_graph_config",{})}
                    sample_graph_cfg.update({"velocity_key": cfg["velocity_key"], "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]})
                    sample_graph_type = val_during_train_cfg.get("val_graph_type", graph_type_train)

                    if sample_graph_type == "knn":
                         # Prepare arguments for vtk_to_knn_graph carefully
                         knn_args_sample = {
                             "k_neighbors": sample_graph_cfg["k"],  # Map 'k'
                             "downsample_n": sample_graph_cfg.get("down_n"),
                             "velocity_key": sample_graph_cfg.get("velocity_key", "U"),
                             "noisy_velocity_key_suffix": sample_graph_cfg.get("noisy_velocity_key_suffix", "_noisy"),
                         }
                         g0_sample = vtk_to_knn_graph(sample_path_t0, **knn_args_sample, use_noisy_data=val_use_noisy_for_run) # Removed device
                         g1_sample = vtk_to_knn_graph(sample_path_t1, **knn_args_sample, use_noisy_data=val_use_noisy_for_run) # Removed device
                    else: # full_mesh
                         # For full_mesh, use_noisy_data is not directly passed to vtk_to_fullmesh_graph
                         # as it reads specific keys. The sample_graph_cfg already has velocity_key which might be noisy.
                         # However, if we want to be explicit, we'd need to adjust how velocity_key is chosen for full_mesh here
                         # based on val_use_noisy_for_run. For now, full_mesh relies on the pre-set velocity_key in sample_graph_cfg.
                         # This part might need refinement if full_mesh also needs explicit clean/noisy switching logic here.
                         # Assuming velocity_key in sample_graph_cfg is already correctly "U" or "U_noisy" based on higher level logic if applicable.
                         g0_sample = vtk_to_fullmesh_graph(sample_path_t0, velocity_key=sample_graph_cfg.get("velocity_key", "U"), pressure_key=cfg.get("pressure_key", "p")) # Removed device
                         g1_sample = vtk_to_fullmesh_graph(sample_path_t1, velocity_key=sample_graph_cfg.get("velocity_key", "U"), pressure_key=cfg.get("pressure_key", "p")) # Removed device

                    # Move sample graphs to device before model call
                    g0_sample = g0_sample.to(device)
                    # g1_sample is used for real_sample = g1_sample.x.cpu(), so it can stay on CPU or be moved if needed by other logic
                    # For model(g0_sample), only g0_sample needs to be on device.

                    with torch.no_grad(): pred_sample = model(g0_sample).cpu()
                    real_sample = g1_sample.x.cpu() # True velocity from the target frame

                    tke_pred_s = turbulent_kinetic_energy(pred_sample)
                    tke_real_s = turbulent_kinetic_energy(real_sample)
                    if abs(tke_real_s) > 1e-9:
                        sample_rel_tke_err = abs(tke_pred_s - tke_real_s) / abs(tke_real_s)
                    sample_cos_sim = cosine_similarity_metric(pred_sample, real_sample)
                    log_dict_epoch[f"{model_name}/sample_rel_TKE_err"] = sample_rel_tke_err
                    log_dict_epoch[f"{model_name}/sample_cosine_sim"] = sample_cos_sim
                except Exception as e:
                    print(f"Warning: Could not compute sample physical metrics: {e}")


            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{cfg['epochs']} [{model_name}] completed in {epoch_duration:.2f}s. "
                  f"Train Loss: {train_metrics['total']:.4e}. "
                  f"Val MSE: {val_metrics.get('val_mse', -1):.4e}. "
                  f"LR: {current_lr:.2e}. Patience: {epochs_no_improve}/{cfg['patience']}.")

            # Log to W&B
            if wandb_run:
                wandb_run.log(log_dict_epoch, step=epoch) # Ensure explicit step is used for the main log call

            # Log to CSV
            csv_writer.writerow([
                epoch, model_name,
                train_metrics.get("total", np.nan),
                train_metrics.get("supervised", np.nan),
                train_metrics.get("divergence", np.nan),
                train_metrics.get("navier_stokes_momentum", np.nan),
                train_metrics.get("lbc", np.nan),
                train_metrics.get("histogram", np.nan),
                train_metrics.get("regularization", np.nan),
                train_metrics.get("alpha_final_batch", np.nan),
                train_metrics.get("beta_final_batch", np.nan),
                val_metrics.get("val_mse", np.nan),
                val_metrics.get("val_rmse_mag", np.nan),
                val_metrics.get("val_mse_div", np.nan),
                val_metrics.get("val_cosine_sim", np.nan),
                val_metrics.get("val_mse_x", np.nan),
                val_metrics.get("val_mse_y", np.nan),
                val_metrics.get("val_mse_z", np.nan),
                val_metrics.get("val_perc_points_within_10_rel_err", np.nan),
                val_metrics.get("val_nrmse_vel", np.nan), # Added NRMSE vel to CSV
                val_metrics.get("val_nrmse_p", np.nan),   # Added NRMSE pressure to CSV
                val_metrics.get("val_avg_max_true_vel_mag", np.nan),
                val_metrics.get("val_avg_max_pred_vel_mag", np.nan),
                current_lr,
                sample_rel_tke_err,
                sample_cos_sim
            ])
            csv_file.flush() # Ensure data is written

            # Early stopping
            if epochs_no_improve >= cfg["patience"]:
                print(f"Early stopping for {model_name} at epoch {epoch} due to no improvement for {cfg['patience']} epochs.")
                break

        print(f"Finished training for model: {model_name}")
        print(f"Best validation MSE for {model_name}: {best_val_metric:.4e}")
        print(f"Best model for {model_name} saved at: {best_model_path}")

        # --- Save collected probe data for this model ---
        if all_probes_data_for_model:
            probes_df = pd.DataFrame(all_probes_data_for_model)
            probes_csv_path = run_output_dir / f"{model_name.lower()}_probes_timeseries.csv"
            probes_df.to_csv(probes_csv_path, index=False)
            print(f"Saved probe data for {model_name} to {probes_csv_path}")
            if wandb_run: # Log as artifact
                probes_artifact_name = f"{model_name}_probe_data"
                probes_art = wandb.Artifact(probes_artifact_name, type="dataset")
                probes_art.add_file(str(probes_csv_path))
                wandb_run.log_artifact(probes_art)
                print(f"Logged {probes_csv_path} as W&B artifact: {probes_artifact_name}")


        # --- Post-training Histogram JSD Validation (optional) ---
        hist_val_after_train_cfg = cfg.get("histogram_validation_after_training", {})
        if not args.skip_hist_val_after_train and hist_val_after_train_cfg.get("enabled", True):
            print(f"\nPerforming histogram JSD validation for best {model_name} model...")
            # Load the best saved model state for this validation
            if best_model_path.exists():
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print(f"Loaded best model state from {best_model_path} for JSD validation.")
            else:
                print(f"Warning: Best model checkpoint {best_model_path} not found. Using current model state for JSD validation.")

            # Determine reference data for JSD (typically noisy validation set)
            # This path should be the "ground truth" against which predictions are compared for JSD.
            jsd_ref_data_root = Path(hist_val_after_train_cfg.get("reference_data_root", cfg["noisy_val_root"]))

            # Graph config for on-the-fly predictions during JSD validation
            base_graph_cfg_jsd = {**cfg.get("graph_config",{}), **hist_val_after_train_cfg.get("graph_config_val",{})}
            jsd_graph_cfg = {
                "k_neighbors": base_graph_cfg_jsd["k"], # Map k to k_neighbors
                "downsample_n": base_graph_cfg_jsd.get("down_n"),
                "velocity_key": cfg.get("velocity_key", "U"), # Ensure default
                "noisy_velocity_key_suffix": cfg.get("noisy_velocity_key_suffix", "_noisy"), # Ensure default
                # This 'use_noisy_data' determines if graph_input uses noisy features from vtk_path_ref.
                # Default to True, assuming model was trained on noisy inputs.
                "use_noisy_data": True
            }
            # Allow override from specific JSD validation graph config if provided
            if "use_noisy_data" in hist_val_after_train_cfg.get("graph_config_val", {}):
                jsd_graph_cfg["use_noisy_data"] = hist_val_after_train_cfg["graph_config_val"]["use_noisy_data"]
            if "velocity_key" in hist_val_after_train_cfg.get("graph_config_val", {}): # Allow override for input key
                 jsd_graph_cfg["velocity_key"] = hist_val_after_train_cfg["graph_config_val"]["velocity_key"]


            jsd_graph_type = hist_val_after_train_cfg.get("graph_type_val", graph_type_train)

            # Velocity key for loading the reference time series for JSD comparison
            jsd_vel_key_ref = hist_val_after_train_cfg.get("velocity_key_ref", cfg.get("velocity_key","U") + cfg.get("noisy_velocity_key_suffix","_noisy") )


            hist_run_config = {
                "reference_data_root": str(jsd_ref_data_root),
                # "predicted_data_root": None, # Predictions will be on-the-fly
                "output_dir_jsd": str(hist_val_output_dir), # Subdir within run_output_dir
                "model_name_prefix": f"{model_name}_BestModel",
                "graph_config_val": jsd_graph_cfg,
                "graph_type_val": jsd_graph_type,
                "velocity_key_ref": jsd_vel_key_ref,
                # "velocity_key_pred" not needed for on-the-fly
            }

            # Update global config for JSD nbins if specified differently
            jsd_nbins = hist_val_after_train_cfg.get("nbins_jsd", cfg["nbins"])
            temp_cfg_for_jsd = {**cfg, "nbins": jsd_nbins, "BINS": jsd_nbins}


            histogram_jsd_validation(
                model=model,
                cfg=temp_cfg_for_jsd,
                run_config=hist_run_config,
                device=device,
                wandb_run=wandb_run
            )
        else:
            print(f"Skipping histogram JSD validation for {model_name}.")


    csv_file.close()
    if wandb_run:
        # Log final artifacts like best models and CSV
        for model_name in models_to_train_names:
            best_model_path_final = models_output_dir / f"{model_name.lower()}_best.pth"
            if best_model_path_final.exists():
                 art_model = wandb.Artifact(f"{model_name}_best_checkpoint", type="model")
                 art_model.add_file(str(best_model_path_final))
                 wandb_run.log_artifact(art_model)

        art_metrics = wandb.Artifact("training_run_outputs", type="results")
        art_metrics.add_file(str(csv_log_path))
        if hist_val_output_dir.exists() and any(hist_val_output_dir.iterdir()): # Check if not empty
            art_metrics.add_dir(str(hist_val_output_dir), name="histograms_after_training")
        wandb_run.log_artifact(art_metrics)
        wandb_run.finish()

    print("\nTraining script finished.")
    print(f"All outputs, models, and logs saved in: {run_output_dir}")

if __name__ == "__main__":
    main()
