import gradio as gr
import pandas as pd
import librosa
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
def extract_audio_features(audio_file):
    try:
        if audio_file==None:
            
            return "Please upload a valid file"
        
        sr,y=audio_file
        y = np.asarray(y, dtype=np.float64)
        duration = librosa.get_duration(y=y, sr=sr)

    # Select the portion of the audio corresponding to the first 2 seconds
        duration_to_keep = 2  # Duration to keep in seconds
        samples_to_keep = int(duration_to_keep * sr)
        y_first_2_seconds = y[:samples_to_keep]
        # Load audio file
        
        print(sr)
        print(y)

        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

        # Take mean along columns to create 1-D arrays
        mfcc_mean = mfcc.mean(axis=1)
        tonnetz_mean = tonnetz.mean(axis=1)
        spectral_contrast_mean = spectral_contrast.mean(axis=1)
        chromagram_mean = chromagram.mean(axis=1)

        # Create DataFrame for each file
        df = pd.DataFrame({
            "mfcc": [mfcc_mean.tolist()],
            "tonnetz": [tonnetz_mean.tolist()],
            "spectral_contrast": [spectral_contrast_mean.tolist()],
            "chromagram": [chromagram_mean.tolist()]
        })

        # Split lists into separate columns
        feature_df = pd.concat([df.drop(['mfcc', 'tonnetz', 'spectral_contrast', 'chromagram'], axis=1), 
                                df['mfcc'].apply(pd.Series).add_prefix('mfcc_'), 
                                df['tonnetz'].apply(pd.Series).add_prefix('tonnetz_'), 
                                df['spectral_contrast'].apply(pd.Series).add_prefix('spectral_contrast_'), 
                                df['chromagram'].apply(pd.Series).add_prefix('chromagram_')], axis=1)

        
        os.chdir(r"C:\Users\naile\Downloads")
        feature_df=pd.DataFrame(feature_df)
        file=open("Random Forest.pkl", 'rb')
        model = pickle.load(file)

        # evaluate model 
        y_predict = model.predict(feature_df)
        y_prob = model.predict_proba(feature_df)
        y_prob=y_prob[0][0]
        file.close()
        if y_predict==0:
            y_predict="Human Speech"
        else:
            y_predict="AI Speech"
            y_prob=1-y_prob
        
        y_prob=y_prob*100
        return f"The prediction is {y_predict} and the percent probability of it is {y_prob}"
    except:
        return "An unexpected error occured! Please refresh the page and try again."

# Create a Gradio interface
inputs = gr.inputs.Audio(label="Upload Audio File")
outputs = gr.outputs.Textbox(label="Predicted Label and Probability")
app = gr.Interface(fn=extract_audio_features, inputs=inputs, outputs=outputs)
app.launch(share=False,debug=True)





