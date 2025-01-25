# Deep Learning Crash Course

[![Early Access - Use Code PREORDER for 25% Off](https://img.shields.io/badge/Early%20Access%20Now%20Available-Use%20Code%20PREORDER%20for%2025%25%20Off-orange)](https://nostarch.com/deep-learning-crash-course)  
by Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo, Giovanni Volpe  
No Starch Press, San Francisco (CA), 2025  
ISBN-13: 9781718503922  
[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

---

1. [Dense Neural Networks for Classification](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch01_DNN_classification)  
   Introduces single- and multi-layer perceptrons for classification tasks (e.g., MNIST digit recognition).

2. [Dense Neural Networks for Regression](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch02_DNN_regression)  
   Explores regression problems and digital twins, focusing on continuous-value prediction with multi-layer networks.

3. [Convolutional Neural Networks for Image Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch03_CNN)  
   Covers convolutional neural networks (CNNs) and their application to tasks such as image classification, localization, style transfer, and DeepDream.

4. **Encoders–Decoders for Latent Space Manipulation**  
   Focuses on autoencoders, variational autoencoders, Wasserstein autoencoders, and anomaly detection, enabling data compression and generation.

>   - [**Code 4-1: Denoising Images with a Denoising Encoder-Decoder**]()  
>     Implements a denoising encoder–decoder that cleans noisy microscopy images. It illustrates how to simulate noisy vs. clean pairs, train a small convolutional model to learn noise removal, and verify that the network isn’t simply memorizing one output (checking for mode collapse).
>
>   - [**Code 4-A: Generating Digit Images with Variational Autoencoders**]()  
>     Shows how to implement a variational autoencoder (VAE) on MNIST digits. The encoder outputs a mean and variance for latent variables, and the decoder reconstructs digits from sampled latent points. The notebook demonstrates random sampling to create new digit images and visualizes how the VAE organizes digits in the latent space.
>
>   - [**Code 4-B: Morphing Images with Wasserstein Autoencoders**]()  
>     Explores a Wasserstein autoencoder (WAE) trained on Fashion-MNIST. You’ll see how to decode random latent points to generate clothing/accessory images and how to interpolate (morph) one image into another in latent space, showcasing the smooth transitions that WAEs learn.
>
>   - [**Code 4-C: Detecting ECG Anomalies with an Autoencoder**]()  
>     Uses a 1D convolutional autoencoder to detect abnormal heartbeats in ECG data. It’s trained exclusively on normal ECGs so that reconstruction error (or distance in latent space) highlights anomalies that deviate from learned “normal” patterns. Demonstrates how to set a threshold for detecting anomalous signals and compares two approaches: reconstruction-based vs. neighbor-based in latent space.

5. [U-Nets for Image Transformation](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch05_UNet)  
   Discusses U-Net architectures for image segmentation, cell counting, and various biomedical imaging applications.

6. [Self-Supervised Learning to Exploit Symmetries](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch06_SelfSupervised)  
   Explains how to use unlabeled data and the symmetries symmetries of a problem for improved model performance with an application in particle localization.

7. [Recurrent Neural Networks for Timeseries Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch07_RNN)  
   Uses recurrent neural networks (RNNs), GRUs, and LSTMs to forecast time-dependent data and build a simple text translator.

8. [Attention and Transformers for Sequence Processing](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention)  
   Introduces attention mechanisms, transformer models, and vision transformers (ViT) for natural language processing (NLP) including improved text translation and sentiment analysis, and image classification.

9. [Generative Adversarial Networks for Image Synthesis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN)  
   Demonstrates generative adversarial networks (GAN) training for image generation, domain translation (CycleGAN), and virtual staining in microscopy.

10. [Diffusion Models for Data Representation and Exploration](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch10_Diffusion)  
    Presents denoising diffusion models for generating and enhancing images, including text-to-image synthesis and image super-resolution.

11. [Graph Neural Networks for Relational Data Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  
    Shows how graph neural networks (GNNs) can model graph-structured data (molecules, cell trajectories, physics simulations) using message passing and graph convolutions.

12. [Active Learning for Continuous Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  
    Describes techniques to iteratively select the most informative samples to label, improving model performance efficiently.

13. [Reinforcement Learning for Strategy Optimization](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  
    Explains Q-learning and Deep Q-learning by teaching an agent to master games such as Tetris.

14. [Reservoir Computing for Predicting Chaos](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  
    Covers reservoir computing methods for forecasting chaotic systems such as the Lorenz attractor.

---
