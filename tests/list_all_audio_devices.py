import pyaudio

audio = pyaudio.PyAudio()

# List all available audio devices
for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print((i, dev['name'], dev['maxInputChannels'], dev['maxOutputChannels']))

audio.terminate()