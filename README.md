# U-Net for Image Segmentation

This repository contains an implementation of the U-Net architecture for image segmentation, trained on the Carvana dataset.

## Project Structure

```
├── model.py       # Defines the U-Net model
├── utils.py       # Utility functions for training and evaluation
├── dataset.py     # Custom dataset class for loading images and masks
├── train.py       # Training pipeline
└── README.md      # Project documentation
```

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision albumentations tqdm
```

## Dataset

The project uses the Carvana dataset for segmentation. The dataset should be structured as follows:

```
Data/
│── train_images/
│── train_masks/
│── val_images/
│── val_masks/
```

- `train_images/` contains input images for training.
- `train_masks/` contains corresponding segmentation masks.
- `val_images/` contains input images for validation.
- `val_masks/` contains corresponding validation masks.

## Training

To train the model, run:

```bash
python train.py
```

### Training Parameters

You can modify the training parameters in `train.py`:

- `LEARNING_RATE = 1e-4`
- `BATCH_SIZE = 16`
- `NUM_EPOCHS = 3`
- `IMAGE_HEIGHT = 160`
- `IMAGE_WIDTH = 240`
- `LOAD_MODEL = False` (Set to `True` to load a saved model checkpoint)

## Model Architecture

The model follows the classic U-Net architecture, consisting of an encoder (downsampling) and decoder (upsampling) with skip connections:

- Downsampling path: Convolutional layers followed by max pooling.
- Bottleneck: Intermediate layers connecting downsampling and upsampling paths.
- Upsampling path: Transposed convolutions and concatenation with skip connections.

![U-Net Architecture](U-Net.jpg)

## Evaluation

After training, you can evaluate the model using:

```bash
python train.py
```

The script will:
- Compute accuracy and Dice score on the validation set.
- Save example predictions to `saved_images/`.

## Checkpoints

The model saves training checkpoints after each epoch:

```bash
my_checkpoint.pth.tar
```

To resume training from a saved checkpoint, set `LOAD_MODEL = True` in `train.py`.

## Results

The model outputs binary segmentation masks for the given input images. Example results are saved in `saved_images/`.