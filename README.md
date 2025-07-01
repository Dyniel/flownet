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

## Recent Enhancements (July 2024)

*   **Memory Optimization**:
    *   Resolved potential CUDA OutOfMemoryErrors during training by enabling graph downsampling (via `down_n` in `graph_config`) and implementing activation checkpointing within the GNN models. This allows training with larger graphs/models on memory-constrained GPUs.
*   **Flexible Data Source Selection**:
    *   The main training script (`scripts/2_train_model.py`) now includes a `--data-source` command-line flag. This allows users to easily switch between training and validating on `clean` or `noisy` datasets (default is `noisy`).
    *   Example: `python scripts/2_train_model.py --data-source clean ...`
*   **Enhanced Metrics**:
    *   **Component-wise Velocity MSE**: Validation now includes Mean Squared Error for each velocity component (X, Y, Z), logged as `val_mse_x`, `val_mse_y`, `val_mse_z`.
    *   **Vorticity Magnitude MSE**: Validation includes Mean Squared Error of the vorticity magnitude, logged as `val_mse_vorticity_mag`. This requires the `pyvista` library.
*   **Improved Visualizations & Outputs**:
    *   **Detailed Validation VTKs**: During validation steps in `2_train_model.py`, the script can now save detailed VTK files. These files include `true_velocity`, `predicted_velocity`, `velocity_error_magnitude`, `true_vorticity_magnitude`, and `predicted_vorticity_magnitude` fields. This feature is controlled by the `save_validation_fields_vtk: true` setting under `validation_during_training` in the YAML configuration.
    *   **W&B Sample Image Logging**: A proof-of-concept has been implemented to log a visual comparison (2D slice of true/predicted/error velocity magnitudes) for a single validation sample directly to Weights & Biases during training. This provides a quick visual check of model performance.
*   **Dependency Update**:
    *   `pyvista` is now used for vorticity calculations. Ensure it is installed in your environment (`pip install pyvista`).

## Wizualna Reprezentacja Projektu

Aby lepiej zrozumieć, jak zorganizowany jest projekt i jak przebiegają typowe zadania, poniżej znajdują się diagramy.

### Architektura Komponentów Projektu

Ten diagram pokazuje główne bloki składowe projektu i ich wzajemne powiązania.

```mermaid
graph LR
    subgraph UserInput[\"Dane Wejściowe i Konfiguracja\"]
        direction LR
        rawData[fa:fa-database Raw CFD Data (.vtk series) in data/]
        customConfig[fa:fa-file-code Custom YAML Configs in config/]
        defaultConfig[fa:fa-file-alt Default Config (default_config.yaml) in config/]
    end

    subgraph CoreLogic[\"Główna Biblioteka (src/cfd_gnn)\"]
        direction TB
        dataUtils[fa:fa-cogs data_utils.py<br>(Data Loading, Noise, Graph Building, Dataset)]
        models[fa:fa-brain models.py<br>(GNN Architectures: FlowNet, Gao)]
        losses[fa:fa-calculator losses.py<br>(Loss Functions)]
        training[fa:fa-person-chalkboard training.py<br>(Training & Validation Loops)]
        metrics[fa:fa-chart-line metrics.py<br>(Evaluation Metrics)]
        validation[fa:fa-check-double validation.py<br>(Standalone Validation Utils)]
        utils[fa:fa-tools utils.py<br>(Helpers: Config, Seed, W&B, VTK I/O)]
    end

    subgraph Scripts[\"Skrypty Uruchomieniowe (scripts/)\"]
        direction TB
        script1[fa:fa-magic 1_prepare_noisy_data.py]
        script2[fa:fa-play-circle 2_train_model.py]
        script3a[fa:fa-project-diagram 3a_validate_knn.py]
        script3b[fa:fa-network-wired 3b_validate_full_mesh.py]
        script4[fa:fa-image 4_validate_histograms.py]
        script5[fa:fa-sitemap 5_combined_validation.py]
        scriptRunExp[fa:fa-rocket run_experiments.py]
    end

    subgraph Outputs[\"Wyniki i Artefakty (outputs/)\"]
        direction LR
        noisyData[fa:fa-database Noisy CFD Data in outputs/noisy_data/]\n        runOutputs[fa:fa-folder-open Per-Run Outputs in outputs/RUN_NAME/]\n        subgraph RunSpecific[\"Wewnątrz outputs/RUN_NAME/\"]\n            direction TB\n            trainedModels[fa:fa-save Saved Models (.pth)]\n            logFiles[fa:fa-file-csv Log Files (.csv)]\n            predictionVTKs[fa:fa-file-export Predicted VTKs]\n            jsdHeatmaps[fa:fa-map JSD Heatmaps]\n        end\n        wandb[fa:fa-cloud Weights & Biases (External)]\n    end

    %% Połączenia
    rawData --> script1
    defaultConfig --> Scripts
    customConfig --> Scripts

    script1 --> noisyData

    CoreLogic --> Scripts

    script2 --> runOutputs
    script3a --> runOutputs
    script3b --> runOutputs
    script4 --> runOutputs
    script5 --> runOutputs
    scriptRunExp ----> script2

    noisyData --> script2
    rawData --> script2

    trainedModels --> script3a
    trainedModels --> script3b
    trainedModels --> script5

    runOutputs --> wandb
```

### Przepływ Pracy (Workflow)

Ten diagram ilustruje typową sekwencję kroków wykonywanych podczas pracy z projektem, od przygotowania danych po analizę wyników.

```mermaid
sequenceDiagram
    participant User as fa:fa-user Użytkownik
    participant P1 as fa:fa-magic 1_prepare_noisy_data.py
    participant P2 as fa:fa-play-circle 2_train_model.py
    participant P3a as fa:fa-project-diagram 3a_validate_knn.py
    participant P3b as fa:fa-network-wired 3b_validate_full_mesh.py
    participant P5 as fa:fa-sitemap 5_combined_validation.py
    participant DataDir as fa:fa-database Dane (data/, outputs/noisy_data/)
    participant ConfigFile as fa:fa-file-code Plik Konfiguracyjny (config/*.yaml)
    participant OutputDir as fa:fa-folder-open Wyniki (outputs/RUN_NAME/)
    participant WandB as fa:fa-cloud Weights & Biases

    User->>ConfigFile: (1) Definiuje/Modyfikuje Konfigurację (ścieżki, parametry)
    User->>P1: (2) Uruchamia (opcjonalnie) z --source-dir, --output-dir
    P1->>DataDir: Czyta czyste dane VTK
    P1->>DataDir: Zapisuje zaszumione dane VTK
    Note over User,DataDir: Krok (2) jest potrzebny, jeśli trenujemy na danych zaszumionych i ich nie ma.

    User->>P2: (3) Uruchamia z --config, --run-name, [--data-source], [--models-to-train], etc.
    P2->>ConfigFile: Wczytuje konfigurację
    P2->>DataDir: Wczytuje dane treningowe i walidacyjne (czyste/zaszumione)
    loop Trening i Walidacja co N epok
        P2->>P2: Przetwarzanie danych, budowa grafów
        P2->>P2: Trening modelu (epoka)
        P2->>P2: Walidacja modelu (na zbiorze walidacyjnym)
        P2->>WandB: Loguje metryki, konfigurację, obrazy
        P2->>OutputDir: Zapisuje metryki lokalnie (CSV)
    end
    P2->>OutputDir: Zapisuje najlepszy model (.pth)
    P2->>OutputDir: (Opcjonalnie) Zapisuje predykcje VTK z walidacji podczas treningu

    User->>P3a: (4a) Uruchamia walidację k-NN (po treningu)
    P3a->>OutputDir: Wczytuje wytrenowany model
    P3a->>ConfigFile: Wczytuje konfigurację walidacji
    P3a->>DataDir: Wczytuje dane walidacyjne
    P3a->>P3a: Przeprowadza inferencję, oblicza metryki (k-NN graphs)
    P3a->>OutputDir: Zapisuje predykcje VTK i metryki CSV
    P3a->>WandB: Loguje metryki walidacji

    User->>P3b: (4b) Uruchamia walidację Full Mesh (po treningu)
    P3b->>OutputDir: Wczytuje wytrenowany model
    P3b->>ConfigFile: Wczytuje konfigurację walidacji
    P3b->>DataDir: Wczytuje dane walidacyjne
    P3b->>P3b: Przeprowadza inferencję, oblicza metryki (Full Mesh graphs)
    P3b->>OutputDir: Zapisuje predykcje VTK i metryki CSV
    P3b->>WandB: Loguje metryki walidacji

    User->>P5: (4c) Uruchamia Połączoną Walidację (np. inferencja + JSD)
    P5->>OutputDir: Wczytuje wytrenowany model
    P5->>ConfigFile: Wczytuje konfigurację
    P5->>DataDir: Wczytuje dane walidacyjne
    P5->>P5: Krok 1: Inferencja (jak 3a lub 3b)
    P5->>OutputDir: Zapisuje predykcje VTK
    P5->>P5: Krok 2: Analiza JSD (porównanie histogramów)
    P5->>OutputDir: Zapisuje wyniki JSD (np. heatmaps VTK)
    P5->>WandB: Loguje metryki połączonej walidacji

    Note over User,OutputDir: Użytkownik analizuje wyniki w W&B oraz lokalne pliki w OutputDir.
```

## Przykładowe Scenariusze Użycia (Szybki Start)

Poniżej znajdują się przykłady, jak uruchomić trening i walidację w różnych popularnych konfiguracjach. Zakładają one, że znajdujesz się w głównym katalogu projektu i masz aktywowane środowisko wirtualne.

**Ważne uwagi przed startem:**

*   **Plik konfiguracyjny:** Wiele ustawień (np. ścieżki do głównych zbiorów danych `train_root`, `val_root`, parametry modelu jak `h_dim`, `layers`) jest wczytywanych z pliku YAML (domyślnie `config/default_config.yaml` lub `config/test_config.yaml` dla `run_experiments.py`, albo plik podany przez `--config`). Poniższe flagi CLI mogą nadpisywać wartości z pliku konfiguracyjnego.
*   **Przygotowanie danych zaszumionych:** Jeśli scenariusz zakłada użycie danych zaszumionych (`--data-source noisy` lub jeśli jest to domyślne ustawienie w konfiguracji), upewnij się, że zostały one wcześniej wygenerowane za pomocą skryptu `scripts/1_prepare_noisy_data.py`. Ścieżki do tych danych (`noisy_train_root`, `noisy_val_root`) również powinny być poprawnie ustawione w pliku konfiguracyjnym.
*   **Nazwa przebiegu (`--run-name`):** Zawsze podawaj unikalną i opisową nazwę dla każdego przebiegu. Będzie ona używana do tworzenia katalogu z wynikami w `outputs/` oraz do identyfikacji przebiegu w Weights & Biases.
*   **Model do trenowania (`--models-to-train`):** Określ, który model chcesz trenować, np. `FlowNet` lub `Gao`.

### Scenariusz 1: Trening i walidacja na danych CZYSTYCH

*   **Cel:** Model uczy się i jest oceniany na idealnych danych, bez symulowanego szumu.
*   **Jak:** Użyj flagi `--data-source clean`. Dodatkowo, w pliku konfiguracyjnym YAML, w sekcji `validation_during_training`, ustaw `use_noisy_data: false`.

```bash
python scripts/2_train_model.py \
    --config config/default_config.yaml \
    --run-name training_on_clean_data \
    --models-to-train FlowNet \
    --data-source clean \
    --epochs 100
    # Upewnij się, że w default_config.yaml masz (lub w Twoim pliku --config):
    # validation_during_training:
    #   enabled: true
    #   use_noisy_data: false
```
*Jeśli chcesz, aby powyższe polecenie zadziałało bez modyfikacji pliku YAML, skrypt `2_train_model.py` musiałby mieć dodatkową flagę CLI do kontrolowania `validation_during_training.use_noisy_data` lub ta logika musiałaby być sprytniej powiązana z `--data-source`.*

### Scenariusz 2: Trening i walidacja na danych ZASZUMIONYCH

*   **Cel:** Standardowy przypadek, gdzie model uczy się na danych z dodanym szumem (symulując np. niedokładności pomiarowe) i jest walidowany na podobnie zaszumionym zbiorze.
*   **Jak:** Użyj flagi `--data-source noisy` (jest to często domyślne zachowanie, jeśli flaga nie jest podana, ale zależy to od implementacji w `2_train_model.py` i ustawień w pliku konfiguracyjnym). W pliku YAML, `validation_during_training.use_noisy_data: true`.

```bash
# Upewnij się, że dane zaszumione istnieją w ścieżkach podanych w konfiguracji!
# np. /home/student2/ethz/CFD_Ubend_other_noisy (zgodnie z test_config.yaml)
# Możesz je wygenerować skryptem 1_prepare_noisy_data.py:
# python scripts/1_prepare_noisy_data.py --source-dir /path/to/clean/train --output-dir /path/to/noisy/train [--p-min ...] [--p-max ...]
# python scripts/1_prepare_noisy_data.py --source-dir /path/to/clean/val --output-dir /path/to/noisy/val [--p-min ...] [--p-max ...]

python scripts/2_train_model.py \
    --config config/default_config.yaml \
    --run-name training_on_noisy_data \
    --models-to-train FlowNet \
    --data-source noisy \
    --epochs 100
    # Upewnij się, że w default_config.yaml masz (lub w Twoim pliku --config):
    # validation_during_training:
    #   enabled: true
    #   use_noisy_data: true
```

### Scenariusz 3: Trening na CZYSTYCH, walidacja na ZASZUMIONYCH

*   **Cel:** Sprawdzenie, jak model wytrenowany na idealnych danych radzi sobie w warunkach zaszumionych.
*   **Jak:** `--data-source clean` dla treningu. W YAML, `validation_during_training.use_noisy_data: true` dla walidacji podczas treningu.

```bash
python scripts/2_train_model.py \
    --config config/default_config.yaml \
    --run-name train_clean_val_noisy \
    --models-to-train FlowNet \
    --data-source clean \
    --epochs 100
    # Upewnij się, że w default_config.yaml masz (lub w Twoim pliku --config):
    # validation_during_training:
    #   enabled: true
    #   use_noisy_data: true
    # (oraz że noisy_val_root wskazuje na zaszumione dane walidacyjne)
```

### Scenariusz 4: Trening na ZASZUMIONYCH, walidacja na CZYSTYCH

*   **Cel:** Sprawdzenie, czy model uczony na "trudniejszych" (zaszumionych) danych potrafi dobrze generalizować do idealnych, czystych warunków.
*   **Jak:** `--data-source noisy` dla treningu. W YAML, `validation_during_training.use_noisy_data: false` dla walidacji podczas treningu.

```bash
python scripts/2_train_model.py \
    --config config/default_config.yaml \
    --run-name train_noisy_val_clean \
    --models-to-train FlowNet \
    --data-source noisy \
    --epochs 100
    # Upewnij się, że w default_config.yaml masz (lub w Twoim pliku --config):
    # validation_during_training:
    #   enabled: true
    #   use_noisy_data: false
    # (oraz że val_root wskazuje na czyste dane walidacyjne)
```

### Scenariusz 5: Kontrola Typu Grafu (k-NN vs Full Mesh)

*   **Cel:** Wybór sposobu konstrukcji grafu dla modelu.
*   **Jak:** Głównie przez plik konfiguracyjny YAML. W sekcji `graph_config` (lub podobnej, np. `test_config.yaml` ma `default_graph_type` oraz `graph_config`):
    *   Dla **k-NN**: ustaw `default_graph_type: "knn"`, oraz zdefiniuj `graph_config.k` (liczba sąsiadów) i `graph_config.down_n` (liczba punktów po downsamplingu, `null` lub `0` dla braku downsamplingu).
    *   Dla **Full Mesh**: ustaw `default_graph_type: "full_mesh"`. Parametry `k` i `down_n` są wtedy zwykle ignorowane.

**Przykład (modyfikacja fragmentu pliku YAML, np. `my_custom_config.yaml`):**

```yaml
# ... inne ustawienia ...

default_graph_type: "full_mesh" # lub "knn"

graph_config:
  k: 12          # Istotne dla knn
  down_n: 20000  # Istotne dla knn, null lub 0 dla braku downsamplingu
  # ... inne parametry graph_config

# ... reszta ustawień ...
```

Następnie uruchom trening z tym plikiem konfiguracyjnym:
```bash
python scripts/2_train_model.py --config config/my_custom_config.yaml --run-name training_with_full_mesh --models-to-train FlowNet --epochs 100
```
*Obecnie skrypt `2_train_model.py` nie posiada flag CLI do bezpośredniego przełączania `default_graph_type` czy parametrów `k` i `down_n`. Należy je ustawić w pliku konfiguracyjnym.*

### Uruchamianie Wielu Eksperymentów (za pomocą `scripts/run_experiments.py`)

Skrypt `scripts/run_experiments.py` jest przeznaczony do uruchamiania serii predefiniowanych eksperymentów. Aby użyć go do konkretnego scenariusza z powyższych:

1.  **Zmodyfikuj `scripts/run_experiments.py`**:
    *   Ustaw `BASE_CONFIG_PATH` na plik YAML, który ma większość potrzebnych ustawień (np. odpowiednie ścieżki, `validation_during_training.use_noisy_data`).
    *   W liście `experiments`, pozostaw lub stwórz tylko jeden słownik, który nadpisuje parametry zgodnie z wybranym scenariuszem (np. dodając `"data_source": "clean"` lub modyfikując `loss_config`).

Przykład (fragment `run_experiments.py` dla Scenariusza 1):
```python
# W scripts/run_experiments.py
BASE_CONFIG_PATH = "config/config_for_clean_training.yaml" # Załóżmy, że ten plik ma use_noisy_data: false
DEFAULT_EPOCHS = 100

experiments = [
    {
        "run_name_suffix": "single_clean_experiment",
        "data_source": "clean", # Nadpisanie na wszelki wypadek
        "models_to_train": ["FlowNet"],
        # ... inne potrzebne nadpisania ...
    },
]
# ... reszta skryptu ...
```
Następnie uruchom: `python scripts/run_experiments.py`


## Project Architecture and Workflow

This project is designed to facilitate the training and evaluation of Graph Neural Networks for CFD predictions. The architecture revolves around a core library (`src/cfd_gnn/`), a set of executable scripts (`scripts/`), and configuration files (`config/`).

### Core Components

*   **`config/`**: Contains YAML configuration files.
    *   `default_config.yaml`: Provides default parameters for all aspects of the pipeline, from data paths and graph construction parameters to model hyperparameters and logging settings.
    *   Custom configuration files can be created to manage different experiments. These are loaded by the scripts and can be overridden by command-line arguments.

*   **`data/`**: This directory is intended as the default location for input CFD datasets.
    *   Datasets typically consist of multiple "cases" (e.g., different simulation runs or geometries like `sUbend_011`, `sUbend_012`).
    *   Each case contains a series of VTK files (e.g., `Frame_00_data.vtk`, `Frame_01_data.vtk`) representing snapshots of the flow field over time, usually located under a `CFD/` subdirectory within the case folder.

*   **`outputs/`**: This is the default directory where all generated files are saved (and is typically gitignored).
    *   For each run (e.g., a training run or a validation run), a subdirectory is created (often named after the `run_name`).
    *   Inside a run's directory, you'll find:
        *   Saved model checkpoints (e.g., `flownet_best.pth`).
        *   Log files (e.g., `training_metrics.csv`, detailed per-frame metrics CSVs).
        *   Predicted VTK files generated during validation.
        *   Visualization outputs, such as JSD heatmaps or slice analysis plots.
        *   Weights & Biases logs (if enabled and not stored elsewhere).

*   **`scripts/`**: Contains Python scripts that drive the different stages of the machine learning pipeline. Each script typically corresponds to a specific task:
    *   `1_prepare_noisy_data.py`: Preprocesses raw CFD data by injecting MRI-like noise to velocity fields and/or point positions. This is useful for simulating sensor noise or for data augmentation.
    *   `2_train_model.py`: The main script for training GNN models. It handles data loading, model initialization, the training loop (including periodic validation), metric logging, and checkpoint saving.
    *   `3a_validate_knn.py`: Validates a trained model using k-Nearest Neighbors (k-NN) graphs.
    *   `3b_validate_full_mesh.py`: Validates a trained model using graphs derived directly from the mesh's tetrahedral cell connectivity.
    *   `4_validate_histograms.py`: Performs standalone Jensen-Shannon Divergence (JSD) histogram validation by comparing two sets of VTK data (e.g., ground truth vs. model predictions).
    *   `5_combined_validation.py`: Orchestrates a full validation sequence, typically involving model inference (like `3a` or `3b`) followed by JSD histogram analysis.
    *   `run_experiments.py`: (If present, or as a concept) Can be used to automate running multiple configurations or experiments.

*   **`src/cfd_gnn/`**: This is the core Python library containing all the reusable logic for the project.
    *   `__init__.py`: Makes the directory a Python package.
    *   `data_utils.py`: Handles data loading from VTK files, noise injection logic, graph construction (both k-NN and full mesh), and the `PairedFrameDataset` class used by PyTorch Geometric DataLoaders.
    *   `losses.py`: Defines custom loss functions used during training, such as the supervised MSE loss, a physics-informed divergence loss, and a histogram-based loss component.
    *   `metrics.py`: Implements various evaluation metrics, including Turbulent Kinetic Energy (TKE), Cosine Similarity, Jensen-Shannon Divergence (JSD) for velocity histograms, vorticity calculations, and new slice-based analysis functions.
    *   `models.py`: Contains definitions of the Graph Neural Network architectures (e.g., `FlowNet`, `RotFlowNet/Gao`) and their building blocks like MLP layers and GNN steps.
    *   `training.py`: Implements the core training loop (`train_single_epoch`) and the during-training validation logic (`validate_on_pairs`).
    *   `utils.py`: Provides general helper functions for tasks like configuration loading, setting random seeds, initializing Weights & Biases, managing device (CPU/GPU) selection, and VTK I/O.
    *   `validation.py`: Contains standalone validation utilities, particularly the pipeline for JSD histogram analysis.

### Workflow Overview

The typical workflow in this project can be summarized as follows:

1.  **Data Preparation**:
    *   Place your raw CFD datasets (series of VTK files per case) into a directory (e.g., `data/my_dataset_clean`).
    *   (Optional) If training with noisy data, use `scripts/1_prepare_noisy_data.py` to generate a noisy version of your dataset (e.g., `outputs/noisy_data/my_dataset_noisy`). This script reads clean VTK files, injects configurable noise, and saves new noisy VTK files.
        ```bash
        python scripts/1_prepare_noisy_data.py --source-dir data/my_dataset_clean --output-dir outputs/noisy_data/my_dataset_noisy ...
        ```

2.  **Configuration**:
    *   Modify `config/default_config.yaml` or create a new YAML file (e.g., `config/my_experiment.yaml`) to set data paths, model parameters (like hidden dimensions, number of layers), training parameters (learning rate, batch size, epochs, loss weights, regularization), graph construction details (k for k-NN, downsampling), and W&B settings.

3.  **Model Training**:
    *   Run `scripts/2_train_model.py`, specifying your configuration, a run name, and the models to train.
        ```bash
        python scripts/2_train_model.py --config config/my_experiment.yaml --run-name my_training_run --models-to-train FlowNet ...
        ```
    *   This script will:
        *   Load the specified dataset (e.g., the noisy dataset prepared in step 1 for training, and a corresponding validation set).
        *   Construct graph pairs using `PairedFrameDataset` from `data_utils.py`.
        *   Initialize the specified GNN model(s) from `models.py`.
        *   Run the training loop defined in `training.py`, using losses from `losses.py`.
        *   Periodically evaluate the model on the validation set, calculating metrics from `metrics.py`.
        *   Log all metrics, configuration, and (optionally) sample visualizations to Weights & Biases and local CSV files.
        *   Save the best model checkpoint(s) to the `outputs/<run_name>/models/` directory.

4.  **Model Validation & Analysis**:
    *   After training, use the saved model checkpoint(s) for more detailed validation.
    *   **k-NN Graph Validation**:
        ```bash
        python scripts/3a_validate_knn.py --model-checkpoint outputs/<run_name>/models/flownet_best.pth --model-name FlowNet ...
        ```
    *   **Full Mesh Graph Validation**:
        ```bash
        python scripts/3b_validate_full_mesh.py --model-checkpoint outputs/<run_name>/models/flownet_best.pth --model-name FlowNet ...
        ```
    *   **Combined Validation (Inference + JSD)**:
        ```bash
        python scripts/5_combined_validation.py --model-checkpoint outputs/<run_name>/models/flownet_best.pth --model-name FlowNet ...
        ```
    *   These validation scripts will:
        *   Load the specified validation dataset.
        *   For each frame, construct the appropriate graph type.
        *   Perform inference using the trained model.
        *   Calculate a comprehensive set of metrics (overall, per-case, per-frame, and per-slice if enabled).
        *   Save detailed metrics to CSV files (e.g., `frame_metrics_....csv`, `frame_slice_metrics_....csv`).
        *   Save predicted VTK fields.
        *   Log aggregated metrics and histograms to W&B.

5.  **Results Review**:
    *   Analyze the metrics logged to W&B dashboards.
    *   Inspect the generated CSV files for detailed performance numbers.
    *   Visualize the predicted VTK files in tools like ParaView to qualitatively assess model performance.

This modular structure allows for flexibility in experimentation and makes it easier to extend or modify specific parts of the pipeline.

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
│       ├── metrics.py            # Evaluation metrics (TKE, CosSim, JSD, Vorticity)
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
    pip install pyvista pooch # For vorticity metrics and enhanced visualizations
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
*   The training script `2_train_model.py` supports a `--data-source [noisy|clean]` flag to select the input dataset type.
*   To save detailed validation VTK fields (including error and vorticity), set `save_validation_fields_vtk: true` under `validation_during_training:` in your YAML config.

## Development Notes

*   **Device Management**: Scripts attempt to use CUDA if available and specified ("auto" or "cuda" in config/CLI). CPU is used as a fallback or if specified.
*   **VTK Keys**: Ensure the `velocity_key`, `pressure_key`, `noisy_velocity_key_suffix`, and `predicted_velocity_key` in your configuration match the fields in your VTK files and your desired output.
*   **Error Handling**: Scripts include basic error handling, but complex data issues might require debugging.
*   **Testing**: Consider adding unit tests to the `tests/` directory for core library functions.

## Future Enhancements (Ideas)

*   More sophisticated data augmentation techniques.
*   Support for additional GNN architectures.
*   Hyperparameter optimization scripts using W&B Sweeps.
*   More detailed post-processing and visualization tools (e.g., velocity profile plots at key cross-sections, pressure drop calculations).
*   Integration with workflow management tools (e.g., Snakemake, Nextflow).
