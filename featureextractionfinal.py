import os
import librosa
import pandas as pd
os.chdir(r"D:\Bsc Ds\sem 4\RIDS\csvs")
# Path to the folder containing .wav files
folder_path = r"D:\Bsc Ds\sem 4\RIDS\LJSpeech-1.1\LJSpeech-1.1\wavs"

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through each .wav file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        
        # Load audio file using librosa
        y, sr = librosa.load(file_path)
        
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
            "filename": [filename],
            "mfcc": [mfcc_mean.tolist()],
            "tonnetz": [tonnetz_mean.tolist()],
            "spectral_contrast": [spectral_contrast_mean.tolist()],
            "chromagram": [chromagram_mean.tolist()]
        })
        
        # Append DataFrame to the list
        dfs.append(df)
        

# Concatenate all DataFrames in the list
feature_df = pd.concat(dfs, ignore_index=True)

# Split lists into separate columns
feature_df = pd.concat([feature_df.drop(['mfcc', 'tonnetz', 'spectral_contrast', 'chromagram'], axis=1), 
                        feature_df['mfcc'].apply(pd.Series).add_prefix('mfcc_'), 
                        feature_df['tonnetz'].apply(pd.Series).add_prefix('tonnetz_'), 
                        feature_df['spectral_contrast'].apply(pd.Series).add_prefix('spectral_contrast_'), 
                        feature_df['chromagram'].apply(pd.Series).add_prefix('chromagram_')], axis=1)

# Save to CSV
feature_df.to_csv("human.csv", index=False)
print("Features extracted and saved to '.csv'.")
