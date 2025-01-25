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

3. **Convolutional Neural Networks for Image Analysis**  
   Covers convolutional neural networks (CNNs) and their application to tasks such as image classification, localization, style transfer, and DeepDream.

>   - [**Code 3-1: Implementing Neural Networks in PyTorch**]()  
>     Demonstrates the basics of convolutional neural networks, including defining convolutional layers, activation functions (ReLU), pooling/upsampling layers, and stacking them into a deeper architecture for image transformation or classification tasks. This notebook also illustrates how to use PyTorch in general.
>     
>   - [**Code 3-A: Classifying Blood Smears with a Convolutional Neural Network**]()  
>     Walks you through loading a malaria dataset, preprocessing images, and training a CNN to distinguish parasitized from uninfected blood cells. It also shows how to generate training logs, measure accuracy, plot ROC curves, and visualize CNN heatmaps and activations to confirm the network’s attention on infected regions.
>
>   - [**Code 3-B: Localizing Microscopic Particles with a Convolutional Neural Network**]()  
>     Focuses on regression tasks (rather than classification) by predicting the (x,y) position of a trapped microparticle in noisy microscope images. It demonstrates how to manually annotate data or use simulated data for training, and includes hooking into intermediate activations to understand how the network learns positional features.
>
>   - [**Code 3-C: Creating DeepDreams**]()  
>     Uses a pre-trained VGG16 model to generate surreal DeepDream visuals, where an image is iteratively adjusted via gradient ascent to amplify the features that specific CNN layers have learned. It implements forward hooks to capture layer activations and shows how to produce dream-like images revealing what the model sees.
>
>   - [**Code 3-D: Transferring Image Styles**]()  
>     Implements neural style transfer, blending the higher-level content of one image with the lower-level textures and brush strokes of another (for example, re-painting a microscopy image in the style of a famous artwork). It demonstrates the use of Gram matrices, L-BFGS optimization, and carefully chosen layers to balance content and style.

4. [Encoders–Decoders for Latent Space Manipulation](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch04_AE)  
   Focuses on autoencoders, variational autoencoders, Wasserstein autoencoders, and anomaly detection, enabling data compression and generation.

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
