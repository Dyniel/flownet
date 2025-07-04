#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_create_visualizations.py
--------------------------
Generates visualizations from probed point and slice data CSVs created by
5_combined_validation.py. Uses Plotly for interactive HTML plots.

Example Usage:
    python scripts/7_create_visualizations.py \
        --points-csv path/to/probed_points_data_MODEL_GRAPH.csv \
        --slices-csv path/to/probed_slices_data_MODEL_GRAPH.csv \
        --output-dir path/to/output_visualizations/
"""

import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast # For safe evaluation of string literals
from plotly.subplots import make_subplots

def ensure_output_dir(path: Path):
    """Creates directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def generate_point_visualizations(points_df: pd.DataFrame, output_dir: Path, model_name: str):
    """Generates animated line plots for each unique probed point."""
    if points_df.empty:
        print("Point data is empty. Skipping point visualizations.")
        return

    # Create unique identifier for points based on target coordinates
    points_df['target_point_id'] = points_df.apply(
        lambda row: f"point_Tx{row['target_point_coords'][0]:.3f}_Ty{row['target_point_coords'][1]:.3f}_Tz{row['target_point_coords'][2]:.3f}", axis=1
    )

    for point_id, group in points_df.groupby('target_point_id'):
        print(f"  Generating plot for {point_id}...")
        group = group.sort_values(by='time')

        # Velocity components
        fig_vel = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Vx", "Vy", "Vz"))

        # Vx
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['true_velocity'].apply(lambda v: v[0]), mode='lines', name='True Vx'), row=1, col=1)
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['pred_velocity'].apply(lambda v: v[0] if isinstance(v, list) and len(v)==3 else None), mode='lines', name='Pred Vx'), row=1, col=1)
        # Vy
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['true_velocity'].apply(lambda v: v[1]), mode='lines', name='True Vy'), row=2, col=1)
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['pred_velocity'].apply(lambda v: v[1] if isinstance(v, list) and len(v)==3 else None), mode='lines', name='Pred Vy'), row=2, col=1)
        # Vz
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['true_velocity'].apply(lambda v: v[2]), mode='lines', name='True Vz'), row=3, col=1)
        fig_vel.add_trace(go.Scatter(x=group['time'], y=group['pred_velocity'].apply(lambda v: v[2] if isinstance(v, list) and len(v)==3 else None), mode='lines', name='Pred Vz'), row=3, col=1)

        fig_vel.update_layout(title_text=f"{model_name} - Velocity Components at {point_id}", height=700)
        fig_vel.update_xaxes(title_text="Time (s)", row=3, col=1)
        output_file_vel = output_dir / f"{model_name}_{point_id}_velocity.html"
        fig_vel.write_html(str(output_file_vel))

        # Vorticity Magnitude
        fig_vort = px.line(group, x='time', y=['true_vort_mag', 'pred_vort_mag'],
                           labels={'value': 'Vorticity Magnitude', 'variable': 'Source'},
                           title=f"{model_name} - Vorticity Magnitude at {point_id}")
        output_file_vort = output_dir / f"{model_name}_{point_id}_vorticity.html"
        fig_vort.write_html(str(output_file_vort))

        # Divergence
        fig_div = px.line(group, x='time', y=['true_divergence', 'pred_divergence'],
                          labels={'value': 'Divergence', 'variable': 'Source'},
                          title=f"{model_name} - Divergence at {point_id}")
        output_file_div = output_dir / f"{model_name}_{point_id}_divergence.html"
        fig_div.write_html(str(output_file_div))

    print(f"Point visualizations saved to {output_dir}")


def generate_slice_visualizations(slices_df: pd.DataFrame, output_dir: Path, model_name: str):
    """Generates animated line plots for each unique slice definition (averaged metrics)."""
    if slices_df.empty:
        print("Slice data is empty. Skipping slice visualizations.")
        return

    slices_df['slice_id'] = slices_df.apply(
        lambda row: f"slice_{row['slice_axis']}_pos{row['slice_position']:.3f}_thick{row['slice_thickness']:.3f}", axis=1
    )

    for slice_id, group in slices_df.groupby('slice_id'):
        print(f"  Generating plot for {slice_id}...")
        group = group.sort_values(by='time')

        # Average Velocity Magnitude
        fig_avg_vel = px.line(group, x='time', y=['avg_true_vel_mag', 'avg_pred_vel_mag'],
                               labels={'value': 'Avg. Velocity Magnitude', 'variable': 'Source'},
                               title=f"{model_name} - Avg. Velocity Magnitude on {slice_id}")
        output_file_avg_vel = output_dir / f"{model_name}_{slice_id}_avg_vel_mag.html"
        fig_avg_vel.write_html(str(output_file_avg_vel))

        # Average Vorticity Magnitude
        fig_avg_vort = px.line(group, x='time', y=['avg_true_vort_mag_slice', 'avg_pred_vort_mag_slice'],
                                labels={'value': 'Avg. Vorticity Magnitude', 'variable': 'Source'},
                                title=f"{model_name} - Avg. Vorticity Magnitude on {slice_id}")
        output_file_avg_vort = output_dir / f"{model_name}_{slice_id}_avg_vort_mag.html"
        fig_avg_vort.write_html(str(output_file_avg_vort))

        # Average Divergence
        fig_avg_div = px.line(group, x='time', y=['avg_true_divergence_slice', 'avg_pred_divergence_slice'],
                               labels={'value': 'Avg. Divergence', 'variable': 'Source'},
                               title=f"{model_name} - Avg. Divergence on {slice_id}")
        output_file_avg_div = output_dir / f"{model_name}_{slice_id}_avg_divergence.html"
        fig_avg_div.write_html(str(output_file_avg_div))

    print(f"Slice visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from probed CFD data.")
    parser.add_argument("--points-csv", type=str, help="Path to the CSV file with probed points data.")
    parser.add_argument("--slices-csv", type=str, help="Path to the CSV file with probed slices data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated HTML visualizations.")
    parser.add_argument("--model-name", type=str, default="Model", help="Name of the model for plot titles.")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    ensure_output_dir(output_path)

    print(f"Output directory: {output_path}")

    if args.points_csv:
        points_file = Path(args.points_csv)
        if points_file.exists():
            print(f"Loading point data from: {points_file}")
            try:
                # Handle potential list-like strings in CSV if not parsed correctly by default
                # Example: '[1.0, 2.0, 3.0]' might be read as a string.
                # We'll assume pandas handles simple numerics, but for lists:
                def safe_literal_eval(val):
                    if isinstance(val, str):
                        try:
                            return ast.literal_eval(val)
                        except (ValueError, SyntaxError): # Catch errors if it's not a valid literal
                            return val # Return original string if not a valid literal
                    return val

                # Specify converters for columns that are expected to be lists
                list_cols_points = ['target_point_coords', 'actual_point_coords', 'true_velocity', 'pred_velocity']
                converters_points = {col: safe_literal_eval for col in list_cols_points}

                points_df = pd.read_csv(points_file, converters=converters_points)
                generate_point_visualizations(points_df, output_path, args.model_name)
            except Exception as e:
                print(f"Error processing points CSV {points_file}: {e}")
        else:
            print(f"Points CSV file not found: {points_file}")

    if args.slices_csv:
        slices_file = Path(args.slices_csv)
        if slices_file.exists():
            print(f"Loading slice data from: {slices_file}")
            try:
                slices_df = pd.read_csv(slices_file)
                generate_slice_visualizations(slices_df, output_path, args.model_name)
            except Exception as e:
                print(f"Error processing slices CSV {slices_file}: {e}")
        else:
            print(f"Slices CSV file not found: {slices_file}")

    print("Visualization script finished.")

if __name__ == "__main__":
    main()
