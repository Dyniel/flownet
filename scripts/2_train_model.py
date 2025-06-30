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

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader


# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import PairedFrameDataset, make_frame_pairs
from src.cfd_gnn.models import FlowNet, RotFlowNet # Add other models if created
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
        "--models-to-train", type=str, nargs="+", default=["FlowNet", "Gao"], # Default from original
        choices=["FlowNet", "Gao", "RotFlowNet"], # Allow RotFlowNet as synonym for Gao
        help="Which model architectures to train (e.g., FlowNet Gao)."
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
    # Training data is typically the noisy dataset
    train_data_path = Path(cfg["noisy_train_root"])
    # Validation data for during-training checks (can be noisy or clean)
    val_during_train_cfg = cfg.get("validation_during_training", {})
    val_use_noisy = val_during_train_cfg.get("use_noisy_data", True)
    val_data_path = Path(cfg["noisy_val_root"] if val_use_noisy else cfg["val_root"])

    print(f"Loading training data from: {train_data_path}")
    print(f"Loading validation data (during training) from: {val_data_path}")

    train_pairs = make_frame_pairs(train_data_path)
    val_pairs_during_train = make_frame_pairs(val_data_path)

    if not train_pairs:
        print(f"Error: No training frame pairs found in {train_data_path}. Exiting.")
        sys.exit(1)
    if not val_pairs_during_train and val_during_train_cfg.get("enabled", True):
        print(f"Warning: No validation frame pairs found in {val_data_path} for during-training validation.")
        # Allow continuation if validation is optional or handled differently

    graph_build_config_train = {**cfg.get("graph_config",{}), "velocity_key": cfg["velocity_key"], "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]}
    graph_type_train = cfg.get("default_graph_type", "knn")

    train_dataset = PairedFrameDataset(
        train_pairs, graph_build_config_train, graph_type=graph_type_train,
        use_noisy_data=True, # Training always on noisy data as per original logic
        device=device # Graphs will be moved to device upon getitem
    )
    train_loader = PyGDataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers",0))

    # --- CSV Logger Setup ---
    csv_log_path = run_output_dir / "training_metrics.csv"
    csv_file = open(csv_log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "epoch", "model_name", "train_loss_total", "train_loss_sup", "train_loss_div", "train_loss_hist",
        "val_mse", "val_rmse_mag", "val_mse_div", "lr",
        "sample_rel_tke_err", "sample_cosine_sim" # Optional, from original script
    ]
    csv_writer.writerow(csv_header)
    print(f"Logging training metrics to: {csv_log_path}")

    # --- Models to Train ---
    models_registry = {
        "FlowNet": FlowNet,
        "Gao": RotFlowNet, # Assuming Gao is RotFlowNet
        "RotFlowNet": RotFlowNet
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
        model = models_registry[model_name](model_arch_cfg).to(device)
        optimizer = Adam(model.parameters(), lr=cfg["lr"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg["patience"] // 2, verbose=True) # Adjusted patience for scheduler

        best_val_metric = float('inf')
        epochs_no_improve = 0

        # Per-model state for early stopping and best model saving
        best_model_path = models_output_dir / f"{model_name.lower()}_best.pth"

        for epoch in range(1, cfg["epochs"] + 1):
            epoch_start_time = time.time()

            train_metrics = train_single_epoch(
                model, train_loader, optimizer,
                loss_weights=cfg.get("loss_config",{}).get("weights",{}),
                histogram_bins=cfg.get("loss_config",{}).get("histogram_bins", cfg["nbins"]),
                device=device,
                clip_grad_norm_value=cfg.get("clip_grad_norm")
            )

            current_lr = optimizer.param_groups[0]['lr']
            log_dict_epoch = {
                f"{model_name}/train_loss_total": train_metrics["total"],
                f"{model_name}/train_loss_sup": train_metrics["supervised"],
                f"{model_name}/train_loss_div": train_metrics["divergence"],
                f"{model_name}/train_loss_hist": train_metrics["histogram"],
                f"{model_name}/learning_rate": current_lr,
                "epoch": epoch
            }

            # Perform validation (during training)
            val_metrics = {}
            if val_during_train_cfg.get("enabled", True) and val_pairs_during_train and \
               (epoch % val_during_train_cfg.get("frequency", 1) == 0 or epoch == cfg["epochs"]):

                val_graph_cfg = {**cfg.get("graph_config",{}), **val_during_train_cfg.get("val_graph_config",{})}
                val_graph_cfg.update({"velocity_key": cfg["velocity_key"], "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]})
                val_graph_type = val_during_train_cfg.get("val_graph_type", graph_type_train)

                val_metrics = validate_on_pairs(
                    model, val_pairs_during_train,
                    graph_config=val_graph_cfg,
                    use_noisy_data_for_val=val_use_noisy,
                    device=device,
                    graph_type=val_graph_type
                )
                log_dict_epoch.update({f"{model_name}/{k}": v for k,v in val_metrics.items()})

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
                         g0_sample = vtk_to_knn_graph(sample_path_t0, **sample_graph_cfg, use_noisy_data=val_use_noisy, device=device)
                         g1_sample = vtk_to_knn_graph(sample_path_t1, **sample_graph_cfg, use_noisy_data=val_use_noisy, device=device)
                    else: # full_mesh
                         g0_sample = vtk_to_fullmesh_graph(sample_path_t0, velocity_key=sample_graph_cfg["velocity_key"], pressure_key=cfg["pressure_key"], device=device)
                         g1_sample = vtk_to_fullmesh_graph(sample_path_t1, velocity_key=sample_graph_cfg["velocity_key"], pressure_key=cfg["pressure_key"], device=device)

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
                wandb_run.log(log_dict_epoch)

            # Log to CSV
            csv_writer.writerow([
                epoch, model_name, train_metrics["total"], train_metrics["supervised"],
                train_metrics["divergence"], train_metrics["histogram"],
                val_metrics.get("val_mse", -1), val_metrics.get("val_rmse_mag", -1), val_metrics.get("val_mse_div", -1),
                current_lr, sample_rel_tke_err, sample_cos_sim
            ])
            csv_file.flush() # Ensure data is written

            # Early stopping
            if epochs_no_improve >= cfg["patience"]:
                print(f"Early stopping for {model_name} at epoch {epoch} due to no improvement for {cfg['patience']} epochs.")
                break

        print(f"Finished training for model: {model_name}")
        print(f"Best validation MSE for {model_name}: {best_val_metric:.4e}")
        print(f"Best model for {model_name} saved at: {best_model_path}")

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
            jsd_graph_cfg = {**cfg.get("graph_config",{}), **hist_val_after_train_cfg.get("graph_config_val",{})}
            jsd_graph_cfg.update({"velocity_key": cfg["velocity_key"], "noisy_velocity_key_suffix": cfg["noisy_velocity_key_suffix"]})
            # Model expects input based on how it was trained (e.g. "U_noisy" if use_noisy_data was true for graph creation)
            jsd_graph_cfg["use_noisy_data"] = True # Assume model input is noisy data for prediction

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
