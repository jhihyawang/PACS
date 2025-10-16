# PACS Mammography Classification Experiment

A deep learning experiment for mammography image classification using PyTorch. This project implements various CNN architectures to classify mammography images into different BI-RADS categories.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project focuses on automated mammography classification using deep learning techniques. The system processes four-view mammography images (L-CC, L-MLO, R-CC, R-MLO) and classifies them into different BI-RADS categories (0-6) for clinical decision support.

### Key Features

- **Multi-view Processing**: Handles 4-view mammography images simultaneously
- **Multiple CNN Architectures**: Supports ResNet50, EfficientNet-B0, and EfficientNet-B3 backbones
- **Advanced Models**: Includes CNN+MLP and CNN+ViT architectures
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Training Visualization**: Real-time training progress and result visualization
- **Class Balancing**: Weighted sampling and loss functions for imbalanced datasets

## 📊 Dataset

The dataset contains mammography images organized into 7 categories (Category 0-6) corresponding to different BI-RADS classifications:

- **Training Set**: 33,264 samples
- **Validation Set**: Available in `datasets/val/`
- **Test Set**: Available in `datasets/test/`

Each sample consists of 4 images:
- L-CC (Left Craniocaudal)
- L-MLO (Left Mediolateral Oblique)
- R-CC (Right Craniocaudal)
- R-MLO (Right Mediolateral Oblique)

## 🏗️ Models

### Available Architectures

1. **CNN Backbone** (`model/cnn.py`)
   - ResNet50, EfficientNet-B0, EfficientNet-B3
   - Shared weights across 4 mammography views
   - Global feature extraction

2. **CNN + MLP Classifier** (`model/cnn_with_mlp.py`)
   - CNN backbone + Multi-Layer Perceptron
   - Feature fusion and classification head

3. **CNN + Vision Transformer** (`model/cnn_with_vit.py`)
   - CNN backbone + Vision Transformer
   - Advanced attention mechanisms

## 🚀 Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jhihyawang/PACS.git
   cd PACS/experiment
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

### Dependencies

- PyTorch >= 2.9.0
- torchvision >= 0.24.0
- pandas >= 2.3.3
- numpy >= 2.2.6
- opencv-python >= 4.12.0.88
- pillow >= 12.0.0
- scikit-learn >= 1.7.2
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- tqdm >= 4.67.1

## 📖 Usage

### Training

```bash
python train.py --config config.json
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model`: Model architecture (`cnn`, `cnn_mlp`, `cnn_vit`)
- `--backbone`: CNN backbone (`resnet50`, `efficientnet_b0`, `efficientnet_b3`)

### Testing

```bash
python test.py --checkpoint checkpoints/best_model_acc.pth
```

### Visualization

Generate training plots and analysis:

```bash
python visualize_training.py
```

### Data Preprocessing

Use the Jupyter notebook for data exploration and preprocessing:

```bash
jupyter notebook preprocessing.ipynb
```

## 📁 Project Structure

```
experiment/
├── model/                    # Model architectures
│   ├── cnn.py               # CNN backbone implementations
│   ├── cnn_with_mlp.py      # CNN + MLP classifier
│   └── cnn_with_vit.py      # CNN + Vision Transformer
├── datasets/                # Dataset files
│   ├── train_label.csv      # Training labels
│   ├── val_label.csv        # Validation labels
│   ├── test_label.csv       # Test labels
│   ├── train/               # Training images
│   ├── val/                 # Validation images
│   └── test/                # Test images
├── checkpoints/             # Model checkpoints
│   ├── best_model_acc.pth   # Best accuracy model
│   ├── best_model_loss.pth  # Best loss model
│   └── training_history.json # Training metrics
├── training_result/         # Training visualizations
├── main.py                  # Main entry point
├── train.py                 # Training script
├── test.py                  # Testing script
├── dataloader.py            # Data loading utilities
├── visualize_training.py    # Visualization utilities
├── preprocessing.ipynb      # Data preprocessing notebook
└── pyproject.toml          # Project configuration
```

## 📈 Results

The model performance can be evaluated using various metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results
- **Training Curves**: Loss and accuracy progression

Results are automatically saved in:
- `checkpoints/training_history.json`: Training metrics
- `training_result/training_plots.png`: Visualization plots

## 🔧 Configuration

Training can be customized through various parameters:

```python
config = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 50,
    'model_type': 'cnn_mlp',
    'backbone': 'resnet50',
    'img_size': 512,
    'class_weights': [1.0, 1.2, 1.5, ...],  # For class balancing
    'early_stopping_patience': 10
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the PACS research initiative. Please refer to the main repository for licensing information.

## 📞 Contact

For questions or collaboration opportunities, please contact the repository maintainer.

---

**Note**: This project is designed for research purposes in medical image analysis. Ensure compliance with relevant medical data regulations and ethical guidelines when using this code with real clinical data.
