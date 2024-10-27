import torch
from transformers import pipeline
import pyaudio
import numpy as np
import time
import wave
import webrtcvad
from voicevox import Client
import asyncio
from deep_translator import GoogleTranslator

# Set the output device index, Make sure it's a speaker device.
# If you want to output in a microphone device, consider using a loopback device.
output_device_index = 8
# 0: least aggressive, 3: most aggressive.
voice_detection_sensitivity = 2
# Longer means it will take some time to end the recording after you stop speaking.
audio_capture_length = 6

# Check if ROCm is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Whisper model using Hugging Face Transformers
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny", # Choices: tiny, small, medium, large, largev2
    chunk_length_s=30,
    device=device,
    generate_kwargs={"language": "<|en|>", "task": "transcribe"}
)

audio = pyaudio.PyAudio()

# Input audio stream
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)  # 20ms frames

vad = webrtcvad.Vad()
vad.set_mode(voice_detection_sensitivity)  # 0: least aggressive, 3: most aggressive

translator = GoogleTranslator(source='en', target='ja')

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

def translate_text(text):
    # Use the deep-translator library to translate the text from English to Japanese
    return translator.translate(text)

def is_speech(frame):
    return vad.is_speech(frame, 16000)

async def text_to_speech(text):
    async with Client() as client:
        audio_query = await client.create_audio_query(text, speaker=2)
        audio_data = await audio_query.synthesis(speaker=2)
        return audio_data

def play_audio(audio_data):
    # Open an output audio stream
    output_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, output_device_index= output_device_index)
    
    # Play the audio data
    output_stream.write(audio_data)
    
    # Close the stream
    output_stream.stop_stream()
    output_stream.close()

async def main():
    try:
        while True:
            frames = []
            speech_detected = False

            for _ in range(0, int(16000 / 320 * audio_capture_length)):  # 6 seconds of audio with 20ms frames
                data = stream.read(320)  # Read 20ms frames
                frames.append(data)
                if is_speech(data):
                    speech_detected = True

            if speech_detected:
                audio_data = b''.join(frames)
                
                # Start the timer
                start_time = time.time()
                
                text = transcribe_audio(audio_data)
                
                # Translate the transcribed text to Japanese
                translated_text = translate_text(text)
                
                # End the timer
                end_time = time.time()
                
                # Calculate the duration
                duration = end_time - start_time
                
                print(f"You said: {text}")
                print(f"Translated to Japanese: {translated_text}")
                print(f"Conversion took {duration:.2f} seconds")
                
                # Convert the translated text to speech and play it
                tts_audio = await text_to_speech(translated_text)
                play_audio(tts_audio)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(main())