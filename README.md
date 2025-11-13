# Mastering SAM Prompts: A Large-Scale Empirical Study in Segmentation Refinement for Scientific Imaging

**Published in Transactions on Machine Learning Research (TMLR), 2025**

This repository contains the code and experiments for the paper:

> *Mastering SAM Prompts: A Large-Scale Empirical Study in Segmentation Refinement for Scientific Imaging* (TMLR 2025)  
> by Stephen Price, Danielle L. Cote, and Elke Rundensteiner.

The project systematically examines how visual prompt design (points, boxes, and masks, along with their augmentations) impacts the Segment Anything Model (SAM) when employed as a model-agnostic segmentation refiner, with a focus on scientific and microscopy images. The study covers thousands of prompt configurations across multiple base segmentation models and compares SAM-based refinement to existing refinement approaches.

---

## Installation

Below is a minimal setup to reproduce the SAM-based refinement environment.

### 1. Create and activate a conda environment

```bash
conda create -n PromptEnv python=3.10
conda activate PromptEnv
# Install Python dependencies
pip install matplotlib transformers scikit-image opencv-python timm pandas ultralytics
pip install torch torchvision
pip install pycocotools
# Install Segment Anything (SAM)
cd segment-anything/
pip install -e .
cd ..
# Download SAM checkpoint
mkdir checkpoints
wget -O checkpoints/sam_vit_l.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195
```
## Reproducing Results and Repository Structure

This repository includes all components needed to reproduce and extend the refinement experiments from the paper.

- **Powder dataset and labels**  
  The dataset and annotations used for the powder experiments (derived from DualSight, see citation below) are stored at:  
  `paper_results/datasets/powder/`
  
- **Pretrained powder model weights**  
  We provide the YOLOv8 segmentation model trained on the powder dataset:  
  `paper_results/savedInference/powder_yolov8n-seg.pt`

- **Initial segmentations (cached for refinement)**  
  To perform initial segmentations that are cached for SAM-based refinement, refer to:  
  `paper_results/savedInference/`
  
- **Ablation study results**  
  All ablation summary results can be found here:  
  `paper_results/ablation/refinementResultsSummary.csv`  
  For more detailed information per configuration, see:  
  `paper_results/ablation/ablation_outputs/{x}.json`

- **Running or extending the ablations**  
  To reproduce or extend the ablation experiments, use:  
  `paper_results/ablation/ablate.py`

- **Helper utilities**  
  Core helper functionality for running experiments and interfacing with SAM is provided in:  
  `paper_results/tools.py`  
  `paper_results/sam_helper.py`

## Citation
If you use this code or build upon this work in academic publications, please cite the paper:
```bibtex
@article{price2025mastering,
  title   = {Mastering SAM Prompts: A Large-Scale Empirical Study in Segmentation Refinement for Scientific Imaging},
  author  = {Price, Stephen and Cote, Danielle L. and Rundensteiner, Elke},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  note    = {Published in TMLR},
  url     = {https://openreview.net/forum?id=cWcTQMpqv6}
}
```
Or if you use the powder dataset, please also cite:
```bibtex
@article{price2025dualsight,
  title={DualSight: multi-stage instance segmentation framework for improved precision},
  author={Price, Stephen and Judd, Kiran and Tsaknopoulos, Kyle and Neamtu, Rodica and Cote, Danielle L},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={27521},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```


