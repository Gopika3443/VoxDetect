from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import librosa
import crepe
import nolds
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
app = Flask(__name__)

# Load SVM model
model = joblib.load('newmodel.pkl')
scaler = StandardScaler()



df1 = pd.read_csv('parkinsons.csv')
x=df1.drop(columns=["Name","status"],axis=1)
y=df1["status"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

scaler.fit(x_train)




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
        mdvp_jitter_percentage = np.round(np.mean(np.abs(np.diff(frequency))), 4)

        # Calculate MDVP:Jitter(Abs)
        mdvp_jitter_absolute = np.round(np.mean(np.diff(frequency)), 4)

        # Calculate RDPE (Relative Difference Pitch Estimate)
        reference_pitch = 440  # Hz (A440)
        rdpe_values = np.abs(frequency - reference_pitch) / reference_pitch
        average_rdpe = np.mean(rdpe_values)

        # Extract D2 (Delta Delta)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mean_d2 = np.round(np.mean(delta2_mfcc), 4)

        # Extract PPE (Pitch Period Entropy)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        ppe = np.round(np.sum(magnitudes) / np.max(magnitudes), 4)

        # Calculate Spread 1 (Standard Deviation)
        spread1 = np.round(np.std(y), 4)

        # Calculate Spread 2 (Range)
        spread2 = np.round(np.max(y) - np.min(y), 4)

        # DFA calculation
        dfa = np.round(nolds.dfa(y), 4)

        # Calculate shimmer
        shimmer = calculate_shimmer(y)

        # Extract other features
        shimmer_mean = np.mean(shimmer)
        shimmer_db = np.mean(librosa.amplitude_to_db(shimmer))

        # Calculate Shimmer:DDA
        shimmer_dda = np.round(np.mean(np.abs(shimmer)), 4)

        # Calculate NHR using librosa.effects.hpss
        harmonic, percussive = librosa.effects.hpss(y)
        nhr = np.round(np.mean(percussive / harmonic), 4)

        # Calculate MDVP:Shimmer
        mdvp_shimmer = np.round(np.mean(np.abs(shimmer)), 4)

        # Calculate MDVP:APQ using a different method
        mdvp_apq = np.round(calculate_mdvp_apq(y), 4)

        # Create a dictionary with extracted features

        features = {
                'MDVP:Fo(Hz)': np.round(np.mean(frequency), 4),
                'MDVP:Fhi(Hz)': np.round(np.max(frequency), 4),
                'MDVP:Flo(Hz)': np.round(np.min(frequency), 4),
                'MDVP:Jitter(%)': mdvp_jitter_percentage,
                'MDVP:Jitter(Abs)': mdvp_jitter_absolute,
                'MDVP:RAP': np.round(np.mean(np.abs(np.diff(pitches, n=2))), 4),
                'MDVP:PPQ': np.round(np.mean(np.abs(np.diff(pitches, n=3))), 4),
                'Jitter:DDP': np.round(np.mean(np.abs(np.diff(pitches, n=2))) * 3, 4),
                'MDVP:Shimmer': mdvp_shimmer,
                'MDVP:Shimmer(dB)': np.round(shimmer_db, 4),
                'Shimmer:APQ3': np.round(np.mean(shimmer[:3 * sr // 2]), 4),
                'Shimmer:APQ5': np.round(np.mean(shimmer[:5 * sr // 2]), 4),
                'MDVP:APQ': mdvp_apq,
                'Shimmer:DDA': shimmer_dda,
                'NHR': nhr,
                'HNR': np.round(np.mean(harmonic / percussive), 4),
                'RPDE': np.round(np.mean(librosa.feature.spectral_flatness(y=y)), 4),
                'DFA': dfa,
                'spread1': spread1,
                'spread2': spread2,
                'D2': mean_d2,
                'PPE': ppe,
            }


        # Convert the dictionary to a DataFrame
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(features, index=[0])

        df.to_csv('features.csv',index=False)

        return df

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the uploaded file
            audio_file_path = 'uploaded_audio.wav'
            file.save(audio_file_path)

            # Extract features from the audio file
            features_df = extract_audio_features(audio_file_path)
            features_df.to_csv('features.csv',index=False)
            # getting error X has 1 features, but StandardScaler is expecting 22 features as input. how to fix?


            ip_data_np = np.asarray(features_df)
    
            s_data = scaler.transform(ip_data_np.reshape(1, -1))

            if features_df is not None:
                # Make prediction using SVM model
                prediction = model.predict(s_data)
                return jsonify({'prediction': prediction.tolist()})
            else:
                return jsonify({'error': 'Failed to extract features from audio'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)