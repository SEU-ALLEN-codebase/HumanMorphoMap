# Reference Datasets  
Publicly shareable data resources.

## Directory Structure  
├── DeKock/ # Mohan et al., 10.25493/ZK52-E1B  
└── allen_human_neuromorpho/ # Berg et al., 10.1038/s41586-021-03813-8  

For each dataset, we provide exploratory analysis scripts in **`analyze_allen_features.py`**, which includes:
- Feature distribution visualization
- Layer prediction using key topological features

Additional utility scripts include:
- **`crop_swcs.py`**: Extracts spherical soma-centered subtrees from SWC files
- **`resample_swc.py`**: Converts SWC files to uniform-step skeleton representations
- **`convert_one_point_soma.py`**: Standardizes non-conventional soma representations to single-point format
- **`fetch_meta.py`**: Batch downloads metadata from NeuroMorpho.org for specified datasets
