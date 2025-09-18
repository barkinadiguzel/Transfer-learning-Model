# Transfer-learning
This code uses transfer learning with a pretrained EfficientNet_B0. It freezes the base layers, replaces the classifier for our 3 classes, and trains only that. DataLoaders handle image preprocessing, and the model is trained with cross-entropy loss. Finally, predictions are made on test and custom images with probabilities shown.
