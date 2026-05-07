# GalaxEye Binary Change Detection

## Project Title & Description
This project implements an end-to-end binary change detection pipeline for paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery, developed for the GalaxEye Space AI Research Intern assessment. The solution leverages an Early-Fusion UNet architecture, where the 3-channel EO and 1-channel SAR images are concatenated and processed jointly. To address the significant class imbalance inherent in disaster datasets, the model is trained using Focal Loss. 

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies.

## Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
Place the dataset in your desired directory and update `config.yaml` with the paths. The expected structure is:
```
dataset/
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```
*Note: Target labels will be automatically remapped to binary (0: No-Change, 1: Change) during data loading.*

## Training
To train the model from scratch, simply run:
```bash
python train.py
```
*Hyperparameters such as batch size, learning rate, and epochs can be modified in `config.yaml`.*

## Evaluation
To evaluate the model on the test or validation splits:
```bash
# Evaluate on validation set
python eval.py --config config.yaml --weights best_model.pth --split val

# Evaluate on test set
python eval.py --config config.yaml --weights best_model.pth --split test
```

## Model Weights
*(To be populated after final training run)*
- **Download Checkpoint**: [Link to Google Drive / HuggingFace Hub]

## Results
*(To be populated after final training run)*

| Split | IoU | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| Val   | TBD | TBD       | TBD    | TBD      |
| Test  | TBD | TBD       | TBD    | TBD      |

## Citation / References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
- Dataset provided by GalaxEye Space.
