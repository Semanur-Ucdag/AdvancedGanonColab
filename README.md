# AdvancedGanonColab
Advanced GAN: CelebA Dataset Adaptation with WGAN-GP
This project demonstrates an advanced implementation of a Generative Adversarial Network (GAN) using the Wasserstein GAN with Gradient Penalty (WGAN-GP) technique. The primary objective is to generate high-quality, realistic facial images from the CelebA dataset. The model leverages modern deep learning practices and integrates with Weights & Biases (WandB) for real-time tracking and logging of the training process.

Table of Contents
Overview
Features
Requirements
Dataset
Architecture
Training
Visualization
Results
Future Work
Acknowledgements
Overview
This project implements a WGAN-GP to improve the stability and performance of GANs. By minimizing the Wasserstein distance between real and generated distributions, and adding a gradient penalty to enforce Lipschitz continuity, the model achieves smoother training and higher-quality image generation.

Features
Generative Adversarial Network (GAN):
A generator to produce realistic images from random noise.
A critic (discriminator) to differentiate real images from generated ones.
Wasserstein Loss with Gradient Penalty (WGAN-GP) for stable and effective training.
Integration with Weights & Biases (WandB) for detailed logging and visualization.
Flexible hyperparameter configuration to experiment with various settings.
Requirements
The following libraries are required to run the code:

Python 3.x
PyTorch
torchvision
PIL (Pillow)
tqdm
matplotlib
numpy
wandb
To install all dependencies, run:

bash
Kodu kopyala
pip install torch torchvision tqdm matplotlib numpy wandb
Dataset
The project uses the CelebA Dataset, a large-scale face attributes dataset with over 200,000 celebrity images.

Download Instructions: The dataset is automatically downloaded and extracted within the project using the provided gdown utility and a pre-defined URL.
Preprocessing: Images are resized to 128x128 and normalized to the range [-1, 1] for optimal model performance.
Architecture
Generator
The generator takes a random noise vector as input and transforms it into a high-resolution image using a series of transposed convolutional layers. Each layer includes:

ConvTranspose2d: Upscales the image.
BatchNorm2d: Normalizes features for stability.
ReLU Activation: Promotes non-linearity in the model.
Tanh Activation: Scales output pixel values between [-1, 1].
Critic
The critic evaluates the realism of images. It uses convolutional layers to downsample images, with:

Conv2d: Reduces spatial dimensions.
InstanceNorm2d: Normalizes features for each sample.
LeakyReLU Activation: Allows gradients for small negative values.
Training
Hyperparameters
Epochs: 20
Batch Size: 32
Learning Rate: 1e-4
Latent Dimension: 200
Critic Updates: 5 per generator update
Gradient Penalty
A gradient penalty term enforces the Lipschitz continuity condition required for Wasserstein distance calculation.

Optimizers
Both generator and critic use the Adam optimizer with parameters:

Betas: (0.5, 0.9)
Logging
Training metrics (losses, gradient norms) and visualizations are logged using WandB.

Visualization
Generated images and loss curves are visualized during training:

Generated Images: Displayed every few steps to monitor generator performance.
Loss Curves: Plotted to evaluate the convergence of both generator and critic.
Results
High-Quality Generated Images: The model successfully generates realistic faces.
Stable Training: Thanks to WGAN-GP, training avoids typical GAN pitfalls like mode collapse.
Example of generated images:

python
Kodu kopyala
noise = gen_noise(8, z_dim, device=device)
fake_images = gen(noise)
show(fake_images)
Future Work
Higher Resolution: Extend the architecture to generate images larger than 128x128.
Additional Features: Incorporate style-based generation or conditioning for more control.
Dataset Expansion: Test the model on other datasets to evaluate its generalizability.
Acknowledgements
This project uses the CelebA dataset and builds on the principles of WGAN-GP. We also thank the WandB platform for providing tools for seamless experiment tracking.


