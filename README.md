# Comparative Study of Generative Models

This project presents an empirical comparison of six generative models
using the Fashion-MNIST dataset. The experiments explore image
reconstruction, image generation, latent space interpolation, and
training behavior.

## Models

The following generative models are implemented and compared:

1. Vanilla Autoencoder (AE)
2. Sparse Autoencoder
3. Variational Autoencoder (VAE)
4. Vanilla Generative Adversarial Network (GAN)
5. Deep Convolutional GAN (DCGAN)
6. Denoising Diffusion Probabilistic Model (DDPM)

A Multi-Layer Perceptron (MLP) classifier is also implemented to
investigate the limitations of using a discriminative classifier
for image generation.

## Dataset

**Fashion-MNIST**

- 28 x 28 grayscale images
- 10 clothing categories
- Training, validation, and testing split: 80%, 10%, and 10%

The dataset is downloaded automatically through `torchvision`
when running the notebook.

## Experiments

The experiments cover:

- Image reconstruction using AE, Sparse AE, and VAE
- Random image generation using VAE, GAN, DCGAN, and DDPM
- Latent space interpolation
- Training loss comparison
- DDPM forward and reverse diffusion process
- MLP classifier analysis and visualization

## Project Structure

```text
DeepLearning-TakeHome/
├── checkpoints/
├── results/
├── src/
├── main.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone this repository:

```bash
git clone https://github.com/alezaandria/DeepLearning-TakeHome.git
cd DeepLearning-TakeHome
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Open `main.ipynb` using Jupyter Notebook or Google Colab.
2. Install the required dependencies.
3. Run the notebook cells sequentially.
4. The notebook will download the Fashion-MNIST dataset if it is not
   already available.
5. Model checkpoints and experiment results are stored in their
   respective folders.

## Results

The project includes visual comparisons of:

- Image reconstruction quality
- Randomly generated samples
- Latent space interpolation
- MLP classifier evidence
- Training loss curves and DDPM denoising trajectory

The visual results are available in the `results/` directory.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
