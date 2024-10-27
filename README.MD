# VoiceVoxEngToJpn

VoiceVoxEngToJpn is a Python project that listens to English speech, transcribes it, translates the text to Japanese, and then converts the translated text back to speech. This project leverages several libraries including PyTorch, Hugging Face Transformers, PyAudio, WebRTC VAD, and VoiceVox.

## Features

- **Automatic Speech Recognition (ASR)**: Uses the Whisper model from Hugging Face Transformers to transcribe English speech.
- **Translation**: Translates the transcribed text from English to Japanese using the Deep Translator library.
- **Text-to-Speech (TTS)**: Converts the translated Japanese text to speech using the VoiceVox library.
- **Real-time Processing**: Processes audio in real-time, detecting speech and performing the above operations seamlessly.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/VoiceVoxEngToJpn.git
    cd VoiceVoxEngToJpn
    ```

2. Install the required dependencies:
    ```bash
    pip install torch transformers pyaudio numpy webrtcvad wave deep-translator voicevox-client
    ```

## Usage

1. Ensure your microphone is connected and recognized by your system.
2. Run the main script:
    ```bash
    python main.py
    ```

3. Speak into the microphone. The system will transcribe your speech, translate it to Japanese, and play the translated speech.

## Code Overview

### Main Components

- **ASR Pipeline**: Uses the Whisper model to transcribe audio.
    ```python
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30,
        device=device,
        generate_kwargs={"language": "<|en|>", "task": "transcribe"}
    )
    ```

- **Audio Stream**: Captures audio input using PyAudio.
    ```python
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)
    ```

- **Voice Activity Detection (VAD)**: Detects speech using WebRTC VAD.
    ```python
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    ```

- **Translation**: Translates text from English to Japanese.
    ```python
    translator = GoogleTranslator(source='en', target='ja')
    ```

- **Text-to-Speech**: Converts text to speech using VoiceVox.
    ```python
    async def text_to_speech(text):
        async with Client() as client:
            audio_query = await client.create_audio_query(text, speaker=2)
            audio_data = await audio_query.synthesis(speaker=2)
            return audio_data
    ```

### Main Loop

The main loop captures audio, detects speech, transcribes, translates, and plays the translated speech.
```python
async def main():
    try:
        while True:
            frames = []
            speech_detected = False

            for _ in range(0, int(16000 / 320 * 6)):  # 6 seconds of audio with 20ms frames
                data = stream.read(320)
                frames.append(data)
                if is_speech(data):
                    speech_detected = True

            if speech_detected:
                audio_data = b''.join(frames)
                start_time = time.time()
                text = transcribe_audio(audio_data)
                translated_text = translate_text(text)
                end_time = time.time()
                duration = end_time - start_time
                print(f"You said: {text}")
                print(f"Translated to Japanese: {translated_text}")
                print(f"Conversion took {duration:.2f} seconds")
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
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [WebRTC VAD](https://webrtc.org/)
- [VoiceVox](https://voicevox.hiroshiba.jp/)
- [Deep Translator](https://github.com/nidhaloff/deep-translator)
