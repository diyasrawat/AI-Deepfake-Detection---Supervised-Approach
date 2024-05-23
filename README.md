# AI-Deepfake-Detection-Supervised-Approach
A classification model to identify between human audio and AI generated audio
# DeepFake Audio Detection using Machine Learning

## Project Overview
This project aims to develop an advanced tool for authenticating audio material with precision and reliability. By employing supervised learning techniques, we trained a model to effectively differentiate between human-generated and AI-generated audio sources. The project focuses on extracting intricate features from audio data and leveraging them to build a robust classification system that can reliably identify the origin of the audio.

## Dataset
The project utilized two datasets:

1. **LJ Speech Dataset**: A collection of over 13,100 short audio clips (1-10 seconds each) featuring a single speaker reading non-fiction text, totaling 24 hours of high-fidelity audio. This dataset represents human-generated audio - https://keithito.com/LJ-Speech-Dataset/

3. **WaveFake Dataset**: Consists of 117,983 generated audio clips (175 hours total) showcasing the capabilities of various deep learning architectures used to create audio deepfakes (MelGAN, Parallel WaveGAN, HiFi-GAN). This dataset represents AI-generated audio - https://zenodo.org/records/5642694

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

## Evaluation Metrics
The models were evaluated using the following metrics:

1. **Accuracy**: The proportion of correctly classified data points, representing the overall effectiveness of the model in making accurate predictions.
2. **Precision**: The proportion of predicted positive labels that are actually correct (positive predictive value).
3. **Recall**: The proportion of actual positive labels that are correctly predicted (sensitivity).
4. **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

## GUI
A graphical user interface (GUI) was built using Gradio, a Python library for creating customizable UI components around machine learning models. The GUI allows users to interact with the trained model and authenticate audio files by uploading them and receiving the classification result (human-generated or AI-generated).

## Results and Conclusion
The Random Forest model emerged as the best-performing model, achieving an accuracy of 98%, precision of 0.97, recall of 1.0, and an F1 score of 0.98. These exceptional results demonstrate the model's ability to accurately classify audio sources as either human-generated or AI-generated.

The project successfully developed a reliable tool for audio authentication, contributing to the mitigation of deepfake audio threats and fostering trust in the digital content landscape.

# References
- S. Team, "Reports - Sensity AI," Sensity, Dec. 05, 2023. https://sensity.ai/reports/
- Dessa, "Detecting audio deepfakes with AI - dessa news - medium," Medium, Dec. 12, 2021. [Online]. Available: https://medium.com/dessa-news/detecting-audio-deepfakes-f2edfd8e2b35
- The LJ Speech Dataset (keithito.com)
- WaveFake: A data set to facilitate audio DeepFake detection (zenodo.org)
- Feature extraction — librosa 0.10.1 documentation
- E. Daehnhardt, "Audio Signal Processing with Python's Librosa," Elena's AI and Python Coding Blog, Living With AI daehnhardt.com, Mar. 05, 2023. https://daehnhardt.com/blog/2023/03/05/python-audio-signal-processing-with-librosa/
- https://wwwaltexsoft.com/blog/audio-analysis/
- R. Wells, "Upsampling and Downsampling Imbalanced Data in Python," wellsr.com. https://wellsr.com/python/upsampling-and-downsampling-imbalanced-data-in-python
- Mcubaa, M., Singha, A., Ikuesanb, R. A., & Ventera, H. (2023). The Effect of Deep Learning Methods on Deepfake Audio Detection for Digital Investigation. In M. Mcubaa, A. Singha, R. A. Ikuesanb, & H. Ventera (Eds.), Procedia Computer Science (Vol. 219, pp. 211–219). Elsevier B.V.
- Bird, J.J., Lotfi, A. (Nottingham Trent University, UK). "REAL-TIME DETECTION OF AI-GENERATED SPEECH FOR DEEPFAKE VOICE CONVERSION." In Proceedings of the International Conference on Artificial Intelligence (ICAI), 1st edition. Nottingham, UK: Nottingham University Press, 2023
- A. Hamza et al., "Deepfake Audio Detection via MFCC Features Using Machine Learning," in IEEE Access, vol. 10, pp. 134018-134028, 2022, doi: 10.1109/ACCESS.2022.3231480.
- E. Daehnhardt, "Audio Signal Processing with Python's Librosa," Elena's AI and Python Coding Blog, Living With AI daehnhardt.com, Mar. 05, 2023. https://daehnhardt.com/blog/2023/03/05/python-audio-signal-processing-with-librosa/
