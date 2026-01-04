# Deep Learning Crash Course

<!--
[![Early Access - Use Code PREORDER for 25% Off](https://img.shields.io/badge/Early%20Access%20Now%20Available-Use%20Code%20PREORDER%20for%2025%25%20Off-orange)](https://nostarch.com/deep-learning-crash-course)  
-->
<p align="left">
  <a href="https://nostarch.com/deep-learning-crash-course">
    <img src="DLCC_frontcover.jpg" width="250">
  </a>
</p>
by Giovanni Volpe, Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo  
No Starch Press, San Francisco (CA), 2026  
ISBN-13: 9781718503922  
[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

---

1. [Dense Neural Networks for Classification](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch01_DNN_classification)  

2. [Dense Neural Networks for Regression](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch02_DNN_regression)  

3. [Convolutional Neural Networks for Image Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch03_CNN)  

4. [Encoders–Decoders for Latent Space Manipulation](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch04_AE)  

5. [U-Nets for Image Transformation](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch05_UNet)  

6. [Self-Supervised Learning to Exploit Symmetries](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch06_SelfSupervised)  

7. [Recurrent Neural Networks for Timeseries Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch07_RNN)  

8. [Attention and Transformers for Sequence Processing](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention)  

9. **Generative Adversarial Networks for Image Synthesis**  
   Demonstrates generative adversarial networks (GAN) training for image generation, domain translation (CycleGAN), and virtual staining in microscopy.

>   - [**Code 9-1: Generating New MNIST Digits with a GAN**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN/ec09_1_gan_mnist/gan_mnist.ipynb)  
>     Implements a simple Deep Convolutional GAN (DCGAN) on the MNIST dataset to generate novel handwritten digits. Illustrates how the generator maps random noise vectors into realistic images, while the discriminator learns to distinguish them from real MNIST samples. Includes visualization of loss curves and intermediate samples during training.
>
>   - [**Code 9-A: Generating MNIST Digits On Demand with a Conditional GAN**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN/ec09_A_cgan_mnist/cgan_mnist.ipynb)  
>     Extends the basic MNIST GAN to a conditional GAN (cGAN), enabling you to specify which digit to generate. Shows how to incorporate class labels into both generator and discriminator by concatenating embedding vectors or feature maps, resulting in targeted digit generation (for example, only 7s).
>
>   - [**Code 9-B: Virtually Staining a Biological Tissue with a Conditional GAN**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN/ec09_B_virtual_staining/virtual_staining.ipynb)  
>     Applies cGANs to transform brightfield images of human motor neurons into virtually stained fluorescence images—without using invasive chemical stains. Demonstrates how to train on paired brightfield and fluorescence images (13 z-planes to 3 fluorescence channels) and produce consistent neuron and nucleus stains. Enables faster, less-destructive microscopy in biomedical studies.
>
>   - [**Code 9-C: Converting Microscopy Images with a CycleGAN**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN/ec09_C_cyclegan/cyclegan.ipynb)  
>     Shows how CycleGAN can handle unpaired images in two domains (e.g., holographic vs. brightfield micrographs). The model learns a forward generator and backward generator with cycle consistency, ensuring that a transformed image can be mapped back to the original domain. Illustrates conversion between holograms and brightfield images even though paired training samples do not exist.

10. [Diffusion Models for Data Representation and Exploration](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch10_Diffusion)  

11. [Graph Neural Networks for Relational Data Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  

12. [Active Learning for Continuous Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  

13. [Reinforcement Learning for Strategy Optimization](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  

14. [Reservoir Computing for Predicting Chaos](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  

---
