# SpinMCIA: Multimodal Integration of scHi-C Data

SpinMCIA is a robust framework designed for the integration of single-cell High-throughput Chromatin Conformation Capture (scHi-C) with other single-cell omics modalities (e.g., RNA-seq, DNA Methylation, ATAC-seq). 

This repository contains the source code, execution scripts, and downstream analysis tutorials required to run SpinMCIA on highly sparse, multidimensional genomic data.

## 1. Installation

We highly recommend using a virtual environment (e.g., `conda` or `venv`). To install the required dependencies, run:

    pip install -r requirements.txt

## 2. Data Availability

Due to file size constraints, sample data is **not** hosted on GitHub. All preprocessed data and pre-computed SpinMCIA outputs are hosted on our [Zenodo Repository (Link to be added)]. You can download this data and place it into any local directory on your machine.

The Zenodo repository contains two primary directories:
* **`Preprocessed_Data/`**: Ready-to-use `.h5ad` files for various datasets.
* **`SpinMCIA_Downstream_Results/`**: Pre-computed SpinMCIA outputs for users who wish to skip the integration step and proceed directly to downstream analysis.

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

👉 **`/tutorials/run_SpinMCIA_example.sh`**

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

For a complete walkthrough on how to analyze the integration outputs (e.g., generating UMAPs, calculating ARI/NMI clustering metrics, and mapping cell types), please see our dedicated Jupyter Notebook tutorial:

👉 **`/tutorials/SpinMCIA_lee_200kb_umap_and_downstream_analysis_tutorial.ipynb`**