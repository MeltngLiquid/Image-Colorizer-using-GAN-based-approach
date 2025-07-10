# Image-Colorizer-using-GAN-based-approach

# 🎨 Colorization of Grayscale Images Using Deep Learning

This project implements a deep learning pipeline to automatically colorize grayscale images using a U-Net-style architecture with a pre-trained VGG16 encoder. The model is trained on the [LHQ-1024](https://github.com/pkuliyi2015/LHQ) dataset and optimized for TPU execution in Google Colab.

---

## 🚀 Features

- **TPU-Optimized Training**: Accelerated training with TensorFlow’s TPU strategy.
- **U-Net with VGG16 Encoder**: Pre-trained VGG16 backbone with skip connections for fine-grained decoding.
- **Custom Decoder**: Multi-stage upsampling with skip connections.
- **LAB Color Space**: Colorization is done in LAB color space for perceptual accuracy.
- **Checkpointing**: Automatic model weight saving based on validation PSNR.
- **Sample Generation**: Saves intermediate colorizations every 5 epochs.
- **Google Drive Integration**: Persistent storage of checkpoints and samples.

---

## 🗂️ Project Structure


---

## 🧠 How It Works

### 🔹 Data Pipeline
- Images are resized and converted from RGB to LAB.
- Only the **L channel** is used as input; the **ab channels** are the target outputs.

### 🔹 Generator
- A U-Net-style model using VGG16 as the encoder.
- Outputs ab channels given the L channel.

### 🔹 Discriminator
- PatchGAN-style discriminator for conditional GAN training.

### 🔹 Training
- Uses adversarial + L1 loss.
- Checkpoints are saved based on validation PSNR.

### 🔹 Inference
- Accepts grayscale images and outputs colorized versions.

---

## 📈 Training Details

| Parameter         | Value                    |
|------------------|--------------------------|
| Batch Size       | Dynamic based on TPUs    |
| Epochs           | 80                       |
| Loss Functions   | BCE + L1                 |
| Validation Metrics | PSNR, SSIM             |

---

## 🛠️ Key Components

- `rgb_to_lab_normalized()` / `lab_to_rgb_normalized()`  
  LAB ↔️ RGB conversions with normalization.
  
- `build_generator()`  
  Constructs the colorization U-Net model.

- `build_discriminator()`  
  PatchGAN-style discriminator.

- `build_dataset()`  
  Loads and prepares dataset for training/validation.

- `train_step()` / `val_step()`  
  TPU-compatible training and validation steps.

- `colorize_image()`  
  Inference on new grayscale images.

---

## ✅ Requirements

- Python 3.x  
- TensorFlow 2.x  
- `tensorflow-io`  
- Google Colab with TPU runtime  
- (Optional) Google Drive for saving outputs

---




