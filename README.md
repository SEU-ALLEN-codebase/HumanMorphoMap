# HumanMorphoMap: Multimodal Data Fusion of Human Cortical Neurons

[![License](https://img.shields.io/badge/apache-2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/bioRxiv-2025.12.26-red.svg)](https://www.biorxiv.org/content/10.64898/2025.12.26.696632v2)

**HumanMorphoMap** is the official code repository for the study:  
*"Multimodal Data Fusion Reveals Morpho-Genetic Variations in Human Cortical Neurons Associated with Tumor Infiltration"*

This framework integrates high-resolution 3D neuronal morphology (reconstructed via **Let'sACT**) with spatial (10x Visium) and bulk transcriptomics to quantify how glioma infiltration reshapes human cortical neurons.

---

## 🚀 Key Features

* **Morphological Profiling:** Automated reconstruction of 3D morphologies of 8,398 human cortical neurons covering various regions.
* **Tumor vs. Normal Comparison:** Statistical pipelines to compare infiltrated tissues vs. normal tissues.
* **Transcriptomic Integration:** Tools to map gene expression gradients (e.g., *CDKN2A*, *TP53*) to morphological phenotypes.
* **Spatial Mapping:** Correlating morphological atrophy with tumor infiltration using spatial transcriptomics spots.

---

## 🛠 Installation

### Prerequisites
* Linux. The source code and dependencies are cross-platform. While currently tested only on Linux, they are expected to work on macOS and Windows.
* Python ≥ 3.9.

### Key Dependencies
* [Vaa3D](https://github.com/Vaa3D) (Visualization and feature calculation. Pre-built Vaa3D-x version 1.1.4)
* [pylib](https://github.com/SEU-ALLEN-codebase/pylib) (Customized Python library for neuron image/morphology processing)
* `scanpy` (Transcriptomic analyses. Tested on version 1.11.2)
* `cell2location` (Spatial deconvolution. Tested on version 0.1.4)
* `numpy`, `pandas`, `scipy` (Core computation. Tested on version 1.26.4, 2.1.1, and 1.11.2 respectively)
* `matplotlib`, `seaborn` (Visualization. Tested on version 3.6.0, 0.13.0)

### Setup Environment
We recommend using `conda` to manage dependencies.

```bash
# Clone the repository
git clone [https://github.com/SEU-ALLEN-codebase/HumanMorphoMap.git](https://github.com/SEU-ALLEN-codebase/HumanMorphoMap.git)
cd HumanMorphoMap

# Create a virtual environment
conda create -n human_morpho python=3.10
conda activate human_morpho
```

### Install dependencies
**1. Internal Library (`pylib`)**
This project depends on our internal library, `pylib`. Please clone the repository and add its location to your `PYTHONPATH`:

```bash
git clone https://github.com/SEU-ALLEN-codebase/pylib.git
export PYTHONPATH=$PYTHONPATH:/path/to/pylib
```

**2. Standard Dependencies**
Install the remaining packages via pip. The installation of dependencies should be within minutes, and it was tested on Ubuntu 20.04 and 24.04.

## User Guide
The overall structure of the project:
```
HumanMorphoMap/
├── meta/                       # Meta processing
├── src/                        # Analytical or visualization utilities. 
├── common_utils/          
├── h01-guided-reconstruction/  # Utilities for EM-based reconstruction optimization
├── human_glioma_CGGA/src/      # Bulk transcriptomics analyses
├── resources/                  # Utilities for processing publicly downloaded morphological datasets
├── soma_morphology/            # Evaluation of soma morphology
├── soma_normalized/            # Post-processing the reconstructons and their features 
├── spatial-enhanced/           # Deprecated
├── spatial_transcript_seu      # Utilities for spatial transcriptomic data processing
├── LICENSE
└── README.md
```

The source code is organized by function and corresponds directly to the figures in the manuscript. To execute an analysis, simply update the file paths under `if __name__ == '__main__'` in the relevant script.

## 📂 Data Availability
* The datasets generated in this study, including automated and manual neuronal reconstructions (`.swc` format) and spatial transcriptomics data, have been deposited on Zenodo (DOI: 10.5281/zenodo.15189542). Comprehensive metadata is available within the repository and in the Supplementary Information accompanying this manuscript.
* Bulk transcriptomic data are downloaded from Chinese Glioma Genome Atlas (CGGA) via https://www.cgga.org.cn/download.jsp.

## 📜 Citation
If you use this code or data in your research, please cite our preprint:
```bash
@article{Liu2025HumanMorphoMap,
  title={Multimodal Data Fusion Reveals Morpho-Genetic Variations in Human Cortical Neurons Associated with Tumor Infiltration},
  author={Yufeng Liu, Zhixi Yun, et al.},
  journal={bioRxiv},
  year={2025},
  doi={10.64898/2025.12.26.696632v2},
  url={[https://www.biorxiv.org/content/10.64898/2025.12.26.696632v2](https://www.biorxiv.org/content/10.64898/2025.12.26.696632v2)}
}
```

