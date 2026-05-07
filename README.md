# GalaxEye Binary Change Detection

## Project Title & Description
This project implements an end-to-end binary change detection pipeline for paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery, developed for the **GalaxEye Space AI Research Intern** assessment. The solution leverages an Early-Fusion UNet architecture, where the 3-channel EO and 1-channel SAR images are concatenated and processed jointly. To address the significant class imbalance inherent in disaster datasets, the model is trained using Focal Loss. 

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
*Note: Target labels are automatically remapped to binary (0: No-Change, 1: Change) dynamically inside the PyTorch Dataloader (`dataset.py`) before training, exactly as mandated by the brief.*

## Training
To train the model from scratch, simply run:
```bash
python train.py
```
*Hyperparameters such as batch size, learning rate, and epochs can be modified in `config.yaml`.*

## Evaluation
To evaluate the model on the test or validation splits and generate the Confusion Matrix and Metrics (IoU, Precision, Recall, F1):
```bash
# Evaluate on validation set
python eval.py --config config.yaml --weights best_model.pth --split val

# Evaluate on test set
python eval.py --config config.yaml --weights best_model.pth --split test
```

## Model Weights
- **Download Checkpoint**: [Click Here to Download best_model.pth](https://github.com/bhukyamadhunaik/galaxeye_change_detection/blob/main/best_model.pth)

## Results

| Split | IoU | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| Val   | 0.634 | 0.782       | 0.765    | 0.773      |
| Test  | 0.612 | 0.765       | 0.748    | 0.756      |

## Citation / References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
- Dataset provided by GalaxEye Space.
