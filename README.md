# AI-Deepfake-Detection-Supervised-Approach
A classification model to identify between human audio and AI generated audio
# DeepFake Audio Detection using Machine Learning

## Project Overview
This project aims to develop an advanced tool for authenticating audio material with precision and reliability. By employing supervised learning techniques, we trained a model to effectively differentiate between human-generated and AI-generated audio sources. The project focuses on extracting intricate features from audio data and leveraging them to build a robust classification system that can reliably identify the origin of the audio.

## Dataset
The project utilized two datasets:

1. **LJ Speech Dataset**: A collection of over 13,100 short audio clips (1-10 seconds each) featuring a single speaker reading non-fiction text, totaling 24 hours of high-fidelity audio. This dataset represents human-generated audio.

2. **WaveFake Dataset**: Consists of 117,983 generated audio clips (175 hours total) showcasing the capabilities of various deep learning architectures used to create audio deepfakes (MelGAN, Parallel WaveGAN, HiFi-GAN). This dataset represents AI-generated audio.

## Process Methodology
1. **Data Collection**: The LJ Speech and WaveFake datasets were obtained, representing human-generated and AI-generated audio samples, respectively.
2. **Data Preparation**: The audio data was prepared for analysis by formatting it as WAV files, a standard format for voice classification.
3. **Feature Extraction**: Spectral features like Mel-Frequency Cepstral Coefficients (MFCCs), Tonal Centroids, Spectral Contrast, and Chromagram were extracted from the audio data using the Python library "librosa".
4. **Data Balancing**: The Synthetic Minority Oversampling Technique (SMOTE) was employed to balance the imbalanced dataset.
5. **Model Training**: Various machine learning models, including Naive Bayes, Logistic Regression, Decision Trees, K-Nearest Neighbors, Random Forest, and XGBoost, were trained on the balanced dataset.
6. **Model Evaluation**: 10-fold cross-validation was performed to evaluate the models and obtain robust performance estimates.

## Models used
The project employed various machine learning techniques, including:

- **Naive Bayes Model (GaussianNB)**
- **Logistic Regression Model**
- **Decision Tree Model**
- **K-Nearest Neighbors (KNN)**
- **Ensemble Models (Random Forest and XGBoost)**

## GUI
A graphical user interface (GUI) was built using Gradio, a Python library for creating customizable UI components around machine learning models. The GUI allows users to interact with the trained model and authenticate audio files by uploading them and receiving the classification result (human-generated or AI-generated).

## Results and Conclusion
The Random Forest model emerged as the best-performing model, achieving an accuracy of 98%, precision of 0.97, recall of 1.0, and an F1 score of 0.98. These exceptional results demonstrate the model's ability to accurately classify audio sources as either human-generated or AI-generated.

The project successfully developed a reliable tool for audio authentication, contributing to the mitigation of deepfake audio threats and fostering trust in the digital content landscape.
