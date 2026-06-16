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

1. [Building and Training Your First Neural Network](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch01_DNN_classification)  

2. [Capturing Trends and Recognizing Patterns with Dense Neural Networks](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch02_DNN_regression)  

3. [Processing Images with Convolutional Neural Networks](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch03_CNN)  

4. [Enhancing, Generating, and Analyzing Data with Autoencoders](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch04_AE)  

5. **Segmenting and Analyzing Images with U-Nets**  
   Discusses U-Net architectures for image segmentation, cell counting, and various biomedical imaging applications.

>   - [**Code 5-1: Segmenting Biological Tissue Images with a U-Net**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch05_UNet/ec05_1_unet/unet.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/DeepLearningCrashCourse/blob/main/Ch05_UNet/ec05_1_unet/unet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
>     Demonstrates how to build and train a U-Net to segment internal cell structures (for example, mitochondria) in electron micrographs. It covers creating pipelines for raw images and labeled masks, using skip connections for detail retention, applying early stopping to avoid overfitting, and evaluating performance via the Jaccard Index (IoU). The notebook also demonstrates data augmentation to improve segmentation robustness.
>
>   - [**Code 5-A: Detecting Quantum Dots in Fluorescence Images with a U-Net**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch05_UNet/ec05_A_qdots_localization/qdots_localization.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/DeepLearningCrashCourse/blob/main/Ch05_UNet/ec05_A_qdots_localization/qdots_localization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
>     Uses a U-Net to localize fluorescent quantum dots in noisy microscopy images. It simulates realistic training data with random positions, intensities, and added noise, and pairs them with masks indicating quantum dot locations. After training on these simulations, the U-Net is tested on real experimental images. You’ll see how accurately it can mark quantum dots by generating centroid-based masks.
>
>   - [**Code 5-B: Counting Cells with a U-Net**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch05_UNet/ec05_B_cell_counting/cell_counting.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/DeepLearningCrashCourse/blob/main/Ch05_UNet/ec05_B_cell_counting/cell_counting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
>     Applies a U-Net to create binary masks of cell nuclei, then uses connected-component labeling to count how many nuclei the mask contains. After simulating or loading real images of stained nuclei, the notebook trains a single-channel output U-Net using a binary cross-entropy loss. Accuracy is measured by comparing predicted cell counts with ground truth, reporting mean absolute and percentage errors. This pipeline automates cell counting and quantifies how close the predictions are to actual counts.

6. [Training Neural Networks with Self-Supervised Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch06_SelfSupervised)  

7. [Processing Time Series and Language with Recurrent Neural Networks](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch07_RNN)  

8. [Processing Language and Classifying Images with Attention and Transformers](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention)  

9. [Creating and Transforming Images with Generative Adversarial Networks](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN)  

10. [Implementing Generative AI with Diffusion Models](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch10_Diffusion)  

11. [Modeling Molecules and Complex Systems with Graph Neural Networks](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  

12. [Continuously Improving Performance with Active Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  

13. [Mastering Decision-Making with Deep Reinforcement Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  

14. [Predicting Chaos with Reservoir Computing](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  

CC. [Companion Examples](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Companion)  

---
