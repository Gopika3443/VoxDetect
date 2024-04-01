
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import librosa
import crepe
import nolds
import warnings

from google.colab import drive
drive.mount('/content/drive')

# Function to calculate jitter
def calculate_jitter_percentage(pitches):
    return np.mean(np.abs(np.diff(pitches)))

def calculate_jitter_absolute(pitches):
    return np.mean(np.diff(pitches))

# Function to calculate shimmer
def calculate_shimmer(y):
    differences = np.diff(np.abs(y))
    return differences

# Function to calculate MDVP:APQ
def calculate_mdvp_apq(y):
    mdvp_var = np.var(y)  # Calculate the variance of the signal related to MDVP
    return mdvp_var

# Function to extract custom audio features
def extract_audio_features(audio_file):
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Crepe pitch detection
        time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)

        # Calculate MDVP:Jitter(%)
        mdvp_jitter_percentage = calculate_jitter_percentage(frequency)

        # Calculate MDVP:Jitter(Abs)
        mdvp_jitter_absolute = calculate_jitter_absolute(frequency)

        # Calculate RDPE (Relative Difference Pitch Estimate)
        reference_pitch = 440  # Hz (A440)
        rdpe_values = np.abs(frequency - reference_pitch) / reference_pitch
        average_rdpe = np.mean(rdpe_values)

        # Extract D2 (Delta Delta)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mean_d2 = np.mean(delta2_mfcc)

        # Extract PPE (Pitch Period Entropy)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        ppe = np.sum(magnitudes) / np.max(magnitudes)

        # Calculate Spread 1 (Standard Deviation)
        spread1 = np.std(y)

        # Calculate Spread 2 (Range)
        spread2 = np.max(y) - np.min(y)

        # DFA calculation
        dfa = nolds.dfa(y)

        # Calculate shimmer
        shimmer = calculate_shimmer(y)

        # Extract other features
        shimmer_mean = np.mean(shimmer)
        shimmer_db = np.mean(librosa.amplitude_to_db(shimmer))

        # Calculate Shimmer:DDA
        shimmer_dda = np.mean(np.abs(shimmer))

        # Calculate NHR using librosa.effects.hpss
        harmonic, percussive = librosa.effects.hpss(y)
        nhr = np.mean(percussive / harmonic)

        # Calculate MDVP:Shimmer
        mdvp_shimmer = np.mean(np.abs(shimmer))

        # Calculate MDVP:APQ using a different method
        mdvp_apq = calculate_mdvp_apq(y)

        # Create a dictionary with extracted features
        features = {
            'MDVP:Fo(Hz)': np.mean(frequency),
            'MDVP:Fhi(Hz)': np.max(frequency),
            'MDVP:Flo(Hz)': np.min(frequency),
            'MDVP:Jitter(%)': mdvp_jitter_percentage,
            'MDVP:Jitter(Abs)': mdvp_jitter_absolute,
            'MDVP:RAP': np.mean(np.abs(np.diff(pitches, n=2))),
            'MDVP:PPQ': np.mean(np.abs(np.diff(pitches, n=3))),
            'Jitter:DDP': np.mean(np.abs(np.diff(pitches, n=2))) * 3,
            'MDVP:Shimmer': mdvp_shimmer,
            'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': np.mean(shimmer[:3 * sr // 2]),
            'Shimmer:APQ5': np.mean(shimmer[:5 * sr // 2]),
            'MDVP:APQ': mdvp_apq,
            'Shimmer:DDA': shimmer_dda,
            'NHR': nhr,
            'HNR': np.mean(harmonic / percussive),
            'RPDE': np.mean(librosa.feature.spectral_flatness(y=y)),
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': mean_d2,
            'PPE': ppe
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(features, index=[0])

        return df

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# Load Parkinson's dataset
df_parkinsons = pd.read_csv("/content/drive/MyDrive/parkinsons.csv")

# Split dataset into features (x) and target (y)
x = df_parkinsons.drop(columns=["name", "status"], axis=1)
y = df_parkinsons["status"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Standardize features
ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

# Train SVM model
model = svm.SVC(kernel="linear")
model.fit(x_train, y_train)

# Evaluate model on training data
x_train_pred = model.predict(x_train)
train_data_acc = accuracy_score(y_train, x_train_pred)
print("Accuracy on training data: ", train_data_acc)

# Evaluate model on testing data
x_test_pred = model.predict(x_test)
test_data_acc = accuracy_score(y_test, x_test_pred)
print("Accuracy on testing data: ", test_data_acc)

# Example usage of feature extraction
audio_file = '/content/drive/MyDrive/pd dataset2/HC_AH/HC_AH/Ah041.wav'
df_audio_features = extract_audio_features(audio_file)

# Ensure feature names match those used during training
df_audio_features.rename(columns={'MDVP:F0(Hz)': 'MDVP:Fo(Hz)', 'Spread1': 'spread1', 'Spread2': 'spread2'}, inplace=True)

# Predict using the extracted features
input_features = ss.transform(df_audio_features)  # Standardize the input features
prediction = model.predict(input_features)

if prediction[0] == 0:
    print("Prediction: No Parkinson's Disease")
else:
    print("Prediction: Yes Parkinson's Disease")

