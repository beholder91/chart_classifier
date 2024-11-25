# Chart Classification Model

This project implements a deep learning model to classify whether an image contains a chart or not. This can be useful for filtering out non-chart images from a large set of images, and focusing on the subtask of chart analysis, such as chart structure extraction, data extraction, etc.

## Project Structure

```
chart_classifier/
├── config.py             # Configuration files
├── data/            # Dataset and data processing
├── models/          # Model architecture
├── trainer/         # Training logic
├── utils/           # Utility functions
├── train.py         # Training script
└── inference.py     # Inference script
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


## Training

To train the model, prepare your data first:
- Place chart images in `data/dataset/charts/`
- Place non-chart images in `data/dataset/non-charts/`

then run:
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