import React, { useState } from 'react';
import './AudioRecorder.css';

const AudioRecorder = ({ onRecordingComplete }) => {
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const handleStartRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const recorder = new MediaRecorder(stream);
        setMediaRecorder(recorder);
        const chunks = [];
        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'audio/wav' });
          setAudioFile(blob);
        };
        recorder.start();
        setIsRecording(true);
      })
      .catch(err => console.error('Error accessing microphone:', err));
  };

  const handleStopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const handleUpload = () => {
    if (audioFile) {
      const formData = new FormData();
      formData.append('file', audioFile);
      fetch('/upload', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json()) // Assuming the response is JSON
      .then(result => {
        // Handle success or display message to the user
        console.log(result);
        // Extract features from the response and do something with them
      })
      .catch(error => {
        console.error('Error uploading file:', error);
        // Handle error or display error message to the user
      });
    }
  };
  

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setAudioFile(file);
  };
  

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Recording</button>
      <button onClick={handleStartRecording} disabled={isRecording}>
        Start Recording
      </button>
      <button onClick={handleStopRecording} disabled={!isRecording}>
        Stop Recording
      </button>
    </div>
  );
};

export default AudioRecorder;
