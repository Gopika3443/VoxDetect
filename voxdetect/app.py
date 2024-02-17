from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'wav'}

# Function to check if a filename has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle file upload and feature extraction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'})

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract features from the uploaded audio file
    features = extract_audio_features(file_path)

    # Check if features are extracted successfully
    if features is not None:
        # Optionally, you can save the features to a CSV file or a database
        # For now, let's return the features as JSON response
        return jsonify({'features': features.to_dict()})
    else:
        return jsonify({'error': 'Failed to extract features'})

if __name__ == '__main__':
    # Define the upload folder
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Run the Flask app
    app.run(debug=True)
