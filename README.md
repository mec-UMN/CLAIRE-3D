# CLAIRE-3D: Composable Chiplet Libraries for AI Inference

This repository contains the AI Library Cloud for CLAIRE-3D, an analytical framework for designing composable, scalable, and reusable chiplet configurations optimized for a broad range of AI inference workloads.

> **Paper:** P. S. Nalla, E. Haque, Y. Liu, S. S. Sapatnekar, J. Zhang, C. Chakrabarti, and K. Cao, "CLAIRE: Composable Chiplet Libraries for AI Inference," in *Proc. Design, Automation and Test in Europe (DATE)*, 2025. DOI: [10.23919/DATE64628.2025.10992960](https://ieeexplore.ieee.org/document/10992960/)

---

## Overview

CLAIRE-3D derives a set of hardened IP and chiplet configurations that are composable, scalable, and reusable by employing an analytical framework trained on a diverse set of AI algorithms — including CNNs, Transformers, LLMs (LLaMA, GPT), and speech models (Whisper). It performs design space exploration (DSE) over systolic array configurations and chiplet-to-chiplet interfaces.

---

## Repository Structure

```
CLAIRE-3D/
├── main_optimize.py           # Main entry point — run this
├── optimize_fns.py            # DSE engine: construct_ind_graphs_sub_fn(), modify_and_run()
├── Postprocess.py             # Postprocessing: Louvain clustering via cluster()
├── Preprocess.py              # Preprocessing utilities
├── noc_fns.py                 # NoC DSE functions
├── nop_fns.py                 # NoP topology DSE: nop_topology_dse()
├── mem_fns.py                 # Memory subsystem functions
├── models.py                  # AI model definitions
├── Cost_Models.py             # Analytical cost models
├── Cost.py                    # Cost computation
├── Utils.py                   # Hardware simulation utilities — references HISIM analytical models (https://github.com/mec-UMN/HiSim)
├── table_utils.py             # Table generation utilities
├── json_utils.py              # JSON parsing utilities
├── test_cycles_cnv_sa.py      # Test script for systolic array cycle estimation
├── PPA_config.json            # PPA configuration
├── PPA_config_comments        # Annotated PPA config
├── params_golden_1.json       # Main configuration file for DSE parameters
├── LICENSE
└── README.md
```

---

## Key Components

| Function | File | Description |
|---|---|---|
| `construct_ind_graphs_sub_fn()` | `optimize_fns.py` | Parses ONNX model graphs into layer-level computation graphs. Note: this script has limited support for dynamic-shape LLM models — see [ONNX Staticalize](#onnx-model-parsing) below for an alternative. |
| `modify_and_run()` | `optimize_fns.py` | Drives the full chiplet DSE across systolic array configurations and chiplet-to-chiplet interfaces |
| `cluster()` | `Postprocess.py` | Louvain community detection for grouping layers onto chiplets |
| `jacc_sim()` | `optimize_fns.py` | Weighted Jaccard similarity for comparing layer computation graphs |
| `nop_topology_dse()` | `nop_fns.py` | Chiplet-to-chiplet topology DSE |

---

## Dependencies

### Python Requirements

- Python 3.8+
- Linux recommended (tested on Ubuntu)

Install dependencies:

```bash
pip install torch onnx onnxruntime networkx numpy scipy pandas matplotlib scikit-learn python-louvain
```

| Package | Purpose |
|---|---|
| `torch` | Model loading and tensor operations |
| `onnx` / `onnxruntime` | ONNX model parsing and inference |
| `networkx` | Computation graph construction |
| `numpy` / `scipy` | Numerical computation |
| `pandas` | Data handling and DSE result tables |
| `matplotlib` | Visualization |
| `scikit-learn` | ML utilities for analytical model |
| `python-louvain` (`community`) | Louvain clustering in postprocessing |

---

## ONNX Model Parsing

`construct_ind_graphs_sub_fn()` in `optimize_fns.py` parses static ONNX graphs into layer-level computation graphs. **This script does not support dynamic-shape LLM ONNX models.**

For dynamic-shape models (e.g., LLaMA, GPT), first staticalize the ONNX model using:

> [https://github.com/chang55245/onnx-model-staticalize](https://github.com/chang55245/onnx-model-staticalize)

Then pass the staticalized model to `construct_ind_graphs_sub_fn()`.

---

## Running

The main entry point is `main_optimize.py`:

```bash
python main_optimize.py
```

Configuration is controlled via `params_golden_1.json` and `PPA_config.json`. The following files may also need to be updated to reflect your target model and DSE settings:

- `main_optimize.py` — top-level run configuration
- `optimize_fns.py` — DSE parameters and search space

Edit these to set:

- Target AI model
- Systolic array size range
- DSE objectives (performance, power, area)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nalla2025claire,
  title     = {{CLAIRE}: Composable Chiplet Libraries for {AI} Inference},
  author    = {Nalla, Pragnya Sudershan and Haque, Emad and Liu, Yaotian and
               Sapatnekar, Sachin S. and Zhang, Jian and Chakrabarti, Chaitali and Cao, Kan},
  booktitle = {Proceedings of the Design, Automation and Test in Europe Conference (DATE)},
  year      = {2025},
  doi       = {10.23919/DATE64628.2025.10992960}
}
```

---

## License

See [LICENSE](LICENSE) for details.
