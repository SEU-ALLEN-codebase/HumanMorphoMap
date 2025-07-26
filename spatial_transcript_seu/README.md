# Spatial Transcriptomics Processing  

## Overview
This directory contains scripts for processing and analyzing spatial transcriptomics data integrated with single-cell RNA-seq references. The pipeline handles data preprocessing, cell type deconvolution, and spatial pattern analysis.

## Scripts and Functions

### 1. Layer Annotation Processing
- **`layer_mapping.py`**:  
  Handles cortical layer-specific annotation mapping for spatial transcriptomics spots.

### 2. Single-cell RNA-seq Processing
- **`scRNA_preprocessing.py`**:  
  - Extracts region-specific cell populations using metadata references (e.g., Sample 83 → Human A44-A45; P117 → MTG)
  - Performs quality filtering with parameters:  
    ```python
    cell_count_cutoff = 5          # Minimum cells per type
    cell_percentage_cutoff = 0.15  # Minimum population percentage
    nonz_mean_cutoff = 2           # Expression threshold
    ```

### 3. Cell Type Deconvolution
- **`predict_cell2loc.py`**:  
  - Trains Cell2Location models to predict cell type composition in spatial spots
  - Extracts pyramidal cell populations from predictions

### 4. Spatial Distribution Analysis
- **`spatial_seu_distance.py`**:  
  Computes spatial distribution patterns and distance relationships between cell types

