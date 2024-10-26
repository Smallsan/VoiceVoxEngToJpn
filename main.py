import torch
from transformers import pipeline
import pyaudio
import numpy as np
import time
import wave
import webrtcvad

# Check if ROCm is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Whisper model using Hugging Face Transformers
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    chunk_length_s=30,
    device=device,
    generate_kwargs={"language": "<|en|>", "task": "transcribe"}
)

audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)  # 20ms frames

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0: least aggressive, 3: most aggressive

print("Listening...")

def save_audio_to_wav(audio_data, filename="temp.wav"):
    # Save the audio data to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(audio_data)

def transcribe_audio(audio_data):
    # Save the audio data to a temporary file
    save_audio_to_wav(audio_data)
    
    # Use the pipeline to transcribe the audio file
    result = pipe("temp.wav")
    return result['text']

def is_speech(frame):
    return vad.is_speech(frame, 16000)

try:
    while True:
        frames = []
        speech_detected = False

        for _ in range(0, int(16000 / 320 * 5)):  # 5 seconds of audio with 20ms frames
            data = stream.read(320)  # Read 20ms frames
            frames.append(data)
            if is_speech(data):
                speech_detected = True

        if speech_detected:
            audio_data = b''.join(frames)
            
            # Start the timer
            start_time = time.time()
            
            text = transcribe_audio(audio_data)
            
            # End the timer
            end_time = time.time()
            
            # Calculate the duration
            duration = end_time - start_time
            
            print(f"You said: {text}")
            print(f"Conversion took {duration:.2f} seconds")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()