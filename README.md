# Chart Classification Model

This project implements a deep learning model to classify whether an image contains a chart or not.

## Project Structure

```
chart_classifier/
├── config/              # Configuration files
├── data/                # Data handling
├── models/              # Model architecture
├── trainer/             # Training logic
├── utils/              # Utility functions
└── inference.py        # Inference script
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/beholder91/chart_classifier.git
cd chart-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
- Place chart images in `data/dataset/charts/`
- Place non-chart images in `data/dataset/non-charts/`

## Training

To train the model:
```bash
python train.py
```

## Inference

To run inference on new images:
```bash
python inference.py
```

## Model Architecture

- Base model: EfficientNet-B0
- Input size: 224x224
- Output: Binary classification (chart/non-chart)

## Requirements

See `requirements.txt` for detailed dependencies.

## License

MIT license