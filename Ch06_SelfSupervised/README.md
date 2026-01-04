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

6. **Self-Supervised Learning to Exploit Symmetries**  
   Explains how to use unlabeled data and the symmetries symmetries of a problem for improved model performance with an application in particle localization.

>   - [**Code 6-1: Localizing Particles Using LodeSTAR**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch06_SelfSupervised/ec06_1_lodestar/lodestar.ipynb)  
>     Demonstrates how to train a self-supervised neural network to determine the sub-pixel position of a particle within a microscope image. The network uses two channels for displacement and one channel for a probability distribution (intensity of detection). This example starts with small, single-particle images and shows how LodeSTAR’s architecture avoids bias by design. You’ll see how the model can accurately predict the x–y position even without direct labels—using only translations (and optionally flips) during training.
>
>   - [**Code 6-A: Localizing Multiple Cells Using LodeSTAR**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch06_SelfSupervised/ec06_A_cell_localization/cell_localization.ipynb)  
>     Applies the LodeSTAR approach to detect multiple mouse stem cells in a brightfield microscopy dataset. Trained solely on a single crop containing one cell, the network can generalize to large frames with many cells. The script showcases how LodeSTAR calculates probability and displacement maps for each pixel, clusters them into detections, and evaluates performance via true centroids provided by Cell Tracking Challenge annotations.

7. [Recurrent Neural Networks for Timeseries Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch07_RNN)  

8. [Attention and Transformers for Sequence Processing](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention)  

9. [Generative Adversarial Networks for Image Synthesis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN)  

10. [Diffusion Models for Data Representation and Exploration](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch10_Diffusion)  

11. [Graph Neural Networks for Relational Data Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  

12. [Active Learning for Continuous Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  

13. [Reinforcement Learning for Strategy Optimization](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  

14. [Reservoir Computing for Predicting Chaos](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  

---
