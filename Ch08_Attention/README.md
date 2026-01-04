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

8. **Attention and Transformers for Sequence Processing**  
   Introduces attention mechanisms, transformer models, and vision transformers (ViT) for natural language processing (NLP) including improved text translation and sentiment analysis, and image classification.

>   - [**Code 8-1: Understanding Attention**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention/ec08_1_attention/attention.ipynb)  
>     Builds attention modules from scratch (dot-product, trainable dot-product, additive) and applies them to toy examples, visualizing attention maps that show which tokens focus on which other tokens. It illustrates how pre-trained embeddings (GloVe) can highlight semantic relationships (like she-her) even without fine-tuning. It clarifies the difference between non-learnable and learnable key/value embeddings.
>
>   - [**Code 8-A: Translating with Attention**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention/ec08_A_nlp_attn/nlp_attn.ipynb)  
>     Improves the seq2seq model from Chapter 7 with dot-product cross-attention to focus on the most relevant parts of source sentences during translation. It demonstrates how attention helps align multi-word phrases and resolves ambiguities. The model surpasses the earlier RNN-based approach by dynamically highlighting crucial source tokens, proving "attention is all you need" for better translations.
>
>   - [**Code 8-B: Performing Sentiment Analysis with a Transformer**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention/ec08_B_transformer/transformer.ipynb)  
>     Implements an encoder-only Transformer using multi-head self-attention and feedforward layers to classify the sentiment of IMDB reviews as positive or negative. Details the entire pipeline: tokenizing, building a vocabulary, batching sequences with masks, stacking multiple Transformer blocks, and adding a dense top for binary classification. The approach yields strong sentiment prediction accuracy, highlighting the parallel processing benefits of Transformers over RNN-based models.
>
>   - [**Code 8-C: Classifying Images with a Vision Transformer**](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch08_Attention/ec08_C_vit/vit.ipynb)  
>     Shows how Transformers can replace convolutions for image tasks by splitting images into patch embeddings. The ViT model is trained from scratch on CIFAR-10, using CutMix to address its weaker inductive biases compared to CNNs. Achieves notable performance when fine-tuned and especially excels if using a pretrained backbone. This underscores ViT’s flexibility and potential to rival or outperform CNNs on visual data.

9. [Generative Adversarial Networks for Image Synthesis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch09_GAN)  

10. [Diffusion Models for Data Representation and Exploration](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch10_Diffusion)  

11. [Graph Neural Networks for Relational Data Analysis](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch11_GNN)  

12. [Active Learning for Continuous Learning](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch12_AL)  

13. [Reinforcement Learning for Strategy Optimization](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch13_RL)  

14. [Reservoir Computing for Predicting Chaos](https://github.com/DeepTrackAI/DeepLearningCrashCourse/tree/main/Ch14_RC)  

---
