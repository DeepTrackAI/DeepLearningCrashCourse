# Deep Learning Crash Course

<!--
[![Early Access - Use Code PREORDER for 25% Off](https://img.shields.io/badge/Early%20Access%20Now%20Available-Use%20Code%20PREORDER%20for%2025%25%20Off-orange)](https://nostarch.com/deep-learning-crash-course)  
-->
<p align="left">
  <a href="https://nostarch.com/deep-learning-crash-course">
    <img src="../DLCC_frontcover.jpg" width="250">
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

9. [Generative Adversarial Networks for Image Synthesis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN)  

10. **Diffusion Models for Data Representation and Exploration**  
    Presents denoising diffusion models for generating and enhancing images, including text-to-image synthesis and image super-resolution.

>   - [**Code 10-1: Generating Digits with a Diffusion Model**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main//Ch10_Diffusion/ec10_1_ddpm_mnist/ddpm_mnist.ipynb)  
>     Implements a Denoising Diffusion Probabilistic Model (DDPM) on MNIST digits. It explains the forward process (adding Gaussian noise at each time step) and the reverse process (a trained denoising U-Net), culminating in random but plausible digit images. It also demonstrates how forward and reverse diffusion steps can be visualized, as well as how different runs from the same noise yield different samples.
>
>   - [**Code 10-A: Generating Bespoke Digits with a Conditional Diffusion Model**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main//Ch10_Diffusion/ec10_A_cddpm_mnist/cddpm_mnist.ipynb)  
>     Extends the DDPM to condition on class labels using classifier-free guidance. Allows specifying which MNIST digit to generate. After training, the network can produce custom digits on demand by blending conditional and unconditional outputs.
>
>   - [**Code 10-B: Generating Images of Digits from Text Prompts**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main//Ch10_Diffusion/ec10_B_text2image/text2image.ipynb)  
>     Demonstrates a mini text-to-image pipeline by pairing a custom transformer encoder (or pretrained CLIP) with a diffusion model. It converts sentences like "There are three horses and two lions. How many lions?" into correct digits. It also showcases classifier-free guidance, adding textual context into an attention U-Net.
>
>   - [**Code 10-C: Generating Super-Resolution Images**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main//Ch10_Diffusion/ec10_C_superresolution/superresolution.ipynb)  
>     Uses a conditional diffusion model to transform low-resolution microscopy images into detailed high-resolution counterparts, showcasing the power of diffusion-based upsampling. It adapts the forward and reverse diffusion to combine the noisy target image with the low-resolution input, effectively learning a mapping to super-resolve biological data.

11. [Graph Neural Networks for Relational Data Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  

12. [Active Learning for Continuous Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  

13. [Reinforcement Learning for Strategy Optimization](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  

14. [Reservoir Computing for Predicting Chaos](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  

---
