import numpy as np
import pandas as pd
import librosa
import crepe
import nolds
import warnings
from werkzeug.utils import secure_filename


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
            'MDVP:F0(Hz)': np.mean(frequency),
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
            'Spread1': spread1,
            'Spread2': spread2,
            'D2': mean_d2,
            'PPE': ppe
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(features, index=[0])


        return df

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# Example usage
audio_file = '/content/drive/MyDrive/pd dataset2/HC_AH/HC_AH/Ah041.wav'
df = extract_audio_features(audio_file)

# Check if df is not None before printing specific columns
if df is not None:
    print(df)
