import opensmile
import pickle
import numpy as np
import pyaudio
import wave
import tempfile
import os
from collections import deque
import time

class RealTimeEmotionRecognizer:
    def __init__(self, model_path='emotion_model_improved.pkl',
                 scaler_path='scaler_improved.pkl',
                 label_encoder_path='label_encoder_improved.pkl',
                 chunk_duration=3.0):
        """
        Real-time emotion recognizer using improved model
        """
        print("Initializing Real-Time Emotion Recognizer...")
        
        # Load model components
        print("Loading model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("✓ Model loaded successfully")
        
        # Initialize openSMILE
        print("Initializing openSMILE...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("✓ openSMILE initialized")
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * chunk_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size)
        self.is_recording = False
        
        # Current emotion tracking
        self.current_emotion = None
        self.current_confidence = 0.0
        self.emotion_history = deque(maxlen=10)  # Track last 10 predictions
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        print("✓ Initialization complete!\n")
    
    def start(self):
        """Start recording and emotion recognition"""
        self.is_recording = True
        
        print("="*60)
        print("REAL-TIME EMOTION RECOGNITION")
        print("="*60)
        print(f"• Analyzing audio in {self.chunk_duration}-second chunks")
        print(f"• Sample rate: {self.sample_rate} Hz")
        print(f"• Emotions: {', '.join(self.label_encoder.classes_)}")
        print("\n🎤 Speak into your microphone...")
        print("Press Ctrl+C to stop\n")
        print("="*60 + "\n")
        
        # Start audio stream
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            while self.is_recording:
                # Read audio
                try:
                    data = self.stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_data)
                    
                    # Process when buffer is full
                    if len(self.audio_buffer) >= self.chunk_size:
                        self._process_chunk()
                        
                except IOError as e:
                    # Handle audio overflow gracefully
                    pass
                    
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("Stopping...")
            self._print_summary()
            self.stop()
    
    def _process_chunk(self):
        """Process audio chunk and predict emotion"""
        # Get audio data
        audio_data = np.array(list(self.audio_buffer), dtype=np.int16)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write WAV file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        try:
            # Extract features
            features = self.smile.process_file(temp_path)
            features_array = features.values[0].reshape(1, -1)
            
            # Normalize and predict
            features_scaled = self.scaler.transform(features_array)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get emotion
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            # Update tracking
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.emotion_history.append(emotion)
            
            # Display result with color coding
            emoji_map = {
                'angry': '😠',
                'calm': '😌',
                'disgust': '🤢',
                'fearful': '😨',
                'happy': '😊',
                'neutral': '😐',
                'sad': '😢',
                'surprised': '😲'
            }
            
            emoji = emoji_map.get(emotion, '🎭')
            
            # Print with timestamp
            timestamp = time.strftime("%H:%M:%S")
            confidence_bar = "█" * int(confidence * 20)
            
            print(f"[{timestamp}] {emoji} {emotion.upper():12s} │ {confidence:.1%} {confidence_bar}", 
                  end='\r', flush=True)
            
        except Exception as e:
            print(f"\nError processing audio: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _print_summary(self):
        """Print summary of emotion detection session"""
        if len(self.emotion_history) > 0:
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total predictions: {len(self.emotion_history)}")
            
            # Count emotions
            emotion_counts = {}
            for emotion in self.emotion_history:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print("\nEmotion distribution:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.emotion_history)) * 100
                bar = "█" * int(percentage / 5)
                print(f"  {emotion:12s} {count:3d} ({percentage:5.1f}%) {bar}")
            
            print("="*60)
    
    def stop(self):
        """Stop recording and clean up"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("✓ Stopped.")

if __name__ == "__main__":
    recognizer = RealTimeEmotionRecognizer()
    recognizer.start()
