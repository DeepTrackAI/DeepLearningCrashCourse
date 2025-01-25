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

4. [Encoders–Decoders for Latent Space Manipulation](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch04_AE)  
   Focuses on autoencoders, variational autoencoders, Wasserstein autoencoders, and anomaly detection, enabling data compression and generation.

5. **U-Nets for Image Transformation**  
   Discusses U-Net architectures for image segmentation, cell counting, and various biomedical imaging applications.

>   - [**Code 5-1: Segmenting Biological Tissue Images with a U-Net**]()  
>     Demonstrates how to build and train a U-Net to segment internal cell structures (for example, mitochondria) in electron micrographs. It covers creating pipelines for raw images and labeled masks, using skip connections for detail retention, applying early stopping to avoid overfitting, and evaluating performance via the Jaccard Index (IoU). The notebook also demonstrates data augmentation to improve segmentation robustness.
>
>   - [**Code 5-A: Detecting Quantum Dots in Fluorescence Images with a U-Net**]()  
>     Uses a U-Net to localize fluorescent quantum dots in noisy microscopy images. It simulates realistic training data with random positions, intensities, and added noise, and pairs them with masks indicating quantum dot locations. After training on these simulations, the U-Net is tested on real experimental images. You’ll see how accurately it can mark quantum dots by generating centroid-based masks.
>
>   - [**Code 5-B: Counting Cells with a U-Net**]()  
>     Applies a U-Net to create binary masks of cell nuclei, then uses connected-component labeling to count how many nuclei the mask contains. After simulating or loading real images of stained nuclei, the notebook trains a single-channel output U-Net using a binary cross-entropy loss. Accuracy is measured by comparing predicted cell counts with ground truth, reporting mean absolute and percentage errors. This pipeline automates cell counting and quantifies how close the predictions are to actual counts.

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
