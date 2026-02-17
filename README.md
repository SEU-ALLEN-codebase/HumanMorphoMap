# HumanMorphoMap: Multimodal Data Fusion of Human Cortical Neurons

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/bioRxiv-2025.12.26-red.svg)](https://www.biorxiv.org/content/10.64898/2025.12.26.696632v2)

**HumanMorphoMap** is the official code repository for the study:  
*> "Multimodal Data Fusion Reveals Morpho-Genetic Variations in Human Cortical Neurons Associated with Tumor Infiltration"*

This framework integrates high-resolution 3D neuronal morphology (reconstructed via **Let'sACT**) with spatial (10x Visium) and bulk transcriptomics to quantify how glioma infiltration reshapes human cortical neurons.

---

## 🚀 Key Features

* **Morphological Profiling:** Automated extraction of 3D features (Soma volume, branch order, tortuosity) from `.swc` files.
* **Tumor vs. Normal Comparison:** Statistical pipelines to compare infiltrated tissues vs. distant normal tissues (paired/unpaired).
* **Transcriptomic Integration:** Tools to map gene expression gradients (e.g., *CDKN2A*, *TP53*) to morphological phenotypes.
* **Spatial Mapping:** Correlating morphological atrophy with distance from the tumor core using spatial transcriptomics spots.

---

## 🛠 Installation

### Prerequisites
* Linux or macOS
* Python ≥ 3.9
* [Vaa3D](https://github.com/Vaa3D) (for visualization and preprocessing)

### Setup Environment
We recommend using `conda` to manage dependencies.

```bash
# Clone the repository
git clone [https://github.com/SEU-ALLEN-codebase/HumanMorphoMap.git](https://github.com/SEU-ALLEN-codebase/HumanMorphoMap.git)
cd HumanMorphoMap

# Create a virtual environment
conda create -n human_morpho python=3.9
conda activate human_morpho

# Install dependencies
pip install -r requirements.txt
# Note: Our internal library 'pylib' is required. 
# If not included, install via: pip install git+[https://github.com/SEU-ALLEN-codebase/pylib.git](https://github.com/SEU-ALLEN-codebase/pylib.git)
