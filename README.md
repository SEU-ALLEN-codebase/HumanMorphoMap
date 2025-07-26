# HumanMorphoMap  

A computational framework for constructing statistical morphology maps of human cortical cells, integrating multi-modal morphological and transcriptomic data.

## Overview  
This project aims to:  
- Create probabilistic morphology atlases of human cortical neurons 
- Correlate cellular morphology with spatial transcriptomics  
- Enable post-tracing guided morphology refinement based on H01 priors using GMM model

## Key Features  
✔️ H01-guided neuronal morphology refinement pipeline  
✔️ Spatial transcriptomics integration  
✔️ Morphological analyses across various conditions 

## Repository Structure  
├── h01-guided-reconstruction/ # H01 dataset-guided morphology refinement
├── meta/ # Project metadata and schemas
├── resources/public_data/ # publicly downloaded morphological datasets
├── spatial_transcript_seu/ # Spatial transcriptomics analysis
├── src/ # Core computational pipelines

## Installation and dependency
- First, download the project
```bash
git clone https://github.com/SEU-ALLEN-codebase/HumanMorphoMap.git
```

- install dependencies
Major dependencies including: Vaa3D-x (version 1.1.4), scanpy (version 1.11.2), and cell2location (version 0.1.4). PyTorch version 2.71.+cu12.6 was utilized for cell2location.


## Usage 
See subfolder READMEs for module-specific instructions.

## License
Apache-2.0 license


