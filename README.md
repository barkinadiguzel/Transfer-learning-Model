# Transfer Learning with EfficientNet_B0

This project demonstrates transfer learning using PyTorch and a pretrained EfficientNet_B0 model. The base layers are frozen, and a new classifier is trained to classify pizza, steak, and sushi images.

## Features

- Uses pretrained EfficientNet_B0 weights from ImageNet.
- Freezes base layers to reduce training time.
- Custom classifier for 3-class prediction.
- Supports training, evaluation, and visualization of predictions.

## Installation

```bash
pip install torch torchvision matplotlib torchinfo
```
## Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
