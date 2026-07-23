# SpinMCIA: Multimodal Integration of scHi-C Data

SpinMCIA is a robust framework designed for the integration of single-cell High-throughput Chromatin Conformation Capture (scHi-C) with other single-cell omics modalities (e.g., RNA-seq, DNA Methylation, ATAC-seq). 

This repository contains the source code, execution scripts, and downstream analysis tutorials required to run SpinMCIA on highly sparse, multidimensional genomic data.

## 1. Installation

We highly recommend using `conda` to create an isolated virtual environment for SpinMCIA. This ensures all dependencies are managed cleanly.

    # Create and activate a new conda environment (Python 3.9+ recommended)
    conda create -n spinmcia_env python=3.9 -y
    conda activate spinmcia_env
    
    # Clone the repository
    git clone https://github.com/ay-lab/SpinMCIA.git
    cd SpinMCIA
    
    # Install the required dependencies
    pip install -r requirements.txt

## 2. Data Availability

Due to file size constraints, sample data is **not** hosted on GitHub. All preprocessed data and pre-computed SpinMCIA outputs are hosted on https://spinmcia.lji.org/data. You can download this data and place it into any local directory on your machine.

The download `.zip` folders contain two primary directories organized to seamlessly support both *de novo* integration and immediate downstream analysis:

*   **`preprocessed_data/`**: Contains ready-to-use `.h5ad` files, feature masks, and metadata required to run the pipeline.
    *   **Resolution Subdirectories (e.g., `100kb/`, `200kb/`, `500kb/`, `1Mb/`):**
        *   **scHi-C AnnData (`*_scHiC.h5ad`):** The preprocessed single-cell Hi-C contact matrices formatted specifically for that genomic resolution.
        *   **Feature Masks (`*_kept_features.npy`):** Available for datasets where sparse columns or bins were filtered out during preprocessing. These arrays are critical during downstream analysis to accurately map the integrated mathematical components back to their original genomic coordinates.
    *   **Secondary Modality Data (e.g., `*_dna_methylation.h5ad`):** The processed count/methylation matrices for the paired omic modality.
    *   **Secondary Feature Masks (`*_kept_features.npy`):** Tracks the retained genomic features for the secondary modality.
    *   **Metadata (`*_metadata.csv`):** Contains the essential cell-level annotations, including cell type labels, used for downstream benchmarking and visualization.

*   **`SpinMCIA_Downstream_Results/`**: Contains pre-computed SpinMCIA outputs for users who wish to skip running SpinMCIA on the provided data and proceed directly to downstream analysis. 
    *   **Resolution Subdirectories (e.g., `100kb/`, `200kb/`, `500kb/`, `1Mb/`):** Each folder mirrors the preprocessed resolutions and contains the exact mathematical outputs of the SpinMCIA pipeline:
        *   **Global Outputs (`global_scores.csv`, `global_loadings.csv`):** The final joint embeddings representing the consensus across modalities. The global scores are used directly for UMAP generation and clustering.
        *   **Modality-Specific Scores (`block_scores_*.csv`):** The low-dimensional representations projected specifically for the scHi-C or secondary modality.
        *   **Feature Loadings (`block_loadings_*.csv`):** The matrix weights that map the embeddings back to the original features. Used in tandem with the `kept_features.npy` arrays to identify the specific genomic loci or contacts driving cell-type identity.
        *   **Metrics & Diagnostics (`block_variances.csv`, `eigvals.csv`, `block_score_weights.csv`):** Statistical outputs detailing the variance explained and modality weighting within the integration.
        *   **Cell Index (`cells_used.txt`):** A list of cell indices retained in the final integration, ensuring alignment with the original metadata.

## 3. Understanding the Input Data

All modalities must be provided to SpinMCIA as `.h5ad` files. The matrices must be flattened such that **rows represent cells** and **columns represent features** (e.g., flattened bin-pairs, genes, or peaks).

**For the scHi-C data specifically:**
* We bin structural contacts at four distinct genomic resolutions: 1 Mb, 500 kb, 200 kb, and 100 kb.
* Only intrachromosomal bin pairs (i,j) where i<j are retained (diagonals are excluded).
* Upper-triangular contact counts are flattened and concatenated across chromosomes into a single 1D feature vector per cell.
* Near-zero bin-pair columns are filtered out to maintain manageable matrix dimensions.

**Crucial Alignment Note:** The cells (rows) must be ordered exactly the same way across all modality files, as illustrated below:

| Modality 1 (scHi-C) | Bin_Pair_1 | Bin_Pair_2 | ... | Bin_Pair_N |
| :--- | :--- | :--- | :--- | :--- |
| **Cell_1** | 0.0 | 2.0 | ... | 0.0 |
| **Cell_2** | 1.0 | 0.0 | ... | 3.0 |

| Modality 2 (RNA-seq) | Gene_1 | Gene_2 | ... | Gene_K |
| :--- | :--- | :--- | :--- | :--- |
| **Cell_1** | 12.4 | 0.0 | ... | 4.2 |
| **Cell_2** | 0.0 | 5.1 | ... | 0.0 |

## 4. Running SpinMCIA

To run the integration pipeline, please refer to the example bash script provided in the repository: 

**`/tutorials/run_SpinMCIA_example.sh`**

This script provides a complete template for running the framework. To use it, simply update the file paths within the script to point to your downloaded `.h5ad` files and define your desired output directory. 

*Note: SpinMCIA is natively equipped for N-modal integration. To run 3+ modalities, simply append the additional `.h5ad` files to the script parameters.*

## 5. Output Interpretation

SpinMCIA generates several `.csv` files inside your specified output folder (these match the files provided in the `SpinMCIA_Downstream_Results/` folder on Zenodo):

* **`global_scores.csv`**: The fully integrated, low-dimensional consensus representation of cells across all modalities (this is the primary file used for UMAP projection and clustering).
* **`global_loadings.csv`**: The weights of the original features contributing to the global components.
* **`block_scores_[Modality].csv`**: The low-dimensional representation of cells specific to an individual modality block.
* **`block_loadings_[Modality].csv`**: The feature weights/contributions derived specifically from that individual modality.
* **`block_score_weights.csv`**: Scalar weights indicating the overall contribution and importance of each modality block to the final global components.
* **`block_variances.csv`**: The variance explained by each specific modality block.
* **`eigvals.csv`**: Eigenvalues associated with the extracted components.

## 6. Downstream Analysis & Tutorials

We provide three dedicated Jupyter Notebook tutorials in the `/tutorials/` directory, tailored to whether you are running the pipeline from scratch, using pre-computed embeddings, or looking for a general template to apply to your own datasets:

* **1. For Freshly Generated Results (Lee 200kb Example):**
  **`/tutorials/Post_SpinMCIA_Run_Tutorial_for_lee_200kb_with_umap_and_downstream_analysis_tutorial_post_spinmcia_run.ipynb`**
  Use this notebook if you just executed `run_SpinMCIA_example.sh` and want to analyze your newly generated Lee 200kb integration outputs (e.g., generating UMAPs, calculating clustering metrics, and mapping cell types).

* **2. For Pre-Computed Embeddings (Lee 200kb Example):**
  **`/tutorials/Provided_SpinMCIA_Embeddings_Tutorial_for_lee_200kb_with_umap_and_downstream_analysis_tutorial.ipynb`**
  Use this notebook if you want to skip running the integration script and jump straight into analyzing our pre-computed Lee 200kb embeddings (downloaded from https://spinmcia.lji.org/data).

* **3. General Downstream Template (For Any Dataset):**
  **`/tutorials/sample_tutorial_without_any_paths.ipynb`**
  A generalized, path-free template designed to teach you the mechanics of SpinMCIA downstream analysis (UMAP visualization, ARI/NMI clustering metrics, etc.). You can easily plug in any newly generated results or pre-computed embeddings into this template to analyze your own custom data.