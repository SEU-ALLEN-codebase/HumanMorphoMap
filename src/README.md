# Core Computational Pipelines  
Main analysis modules for morphology mapping.

## Key scripts
This repository contains both production and exploratory scripts. While many scripts were used for analytical trials, only a subset are required for final production. All scripts in the **`deprecated_scripts`** directory and some in the current directory are maintained for reference purposes.

Key production scripts include:
- **`infiltration_vs_normal.py`**: Morphological comparison between neurons from infiltrated vs. normal tissues
- **`morphology_distance.py`**: Analyzes morphology-distance relationships
- **`merfish_distance.py`**: Analyzes MERFISH-distance relationships
- **`cross_merfish_morphology.py`**: Performs transcriptomic-morphology inter-modality analysis

Note: The 10x Visium vs. spatial distance analysis is located separately in **`../spatial_transcript_seu/spatial_seu_distance.py`**.


