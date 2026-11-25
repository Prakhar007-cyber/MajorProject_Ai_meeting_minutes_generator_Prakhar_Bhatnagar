import numpy as np
import sounddevice as sd
import soundfile as sf

# 16000 samples per second is what Whisper and pyannote expect
SAMPLE_RATE = 16000


# Check that the laptop has at least one microphone. Returns True or False.
def microphone_available():
    try:
        sd.query_devices(kind="input")
        return True
    except Exception:
        return False


# Start recording from the default mic. Returns the stream and a list that fills with audio chunks.
def start_recording():
    audio_chunks = []

    # sounddevice calls this function again and again with small pieces of audio
    def save_chunk(indata, frames, time, status):
        audio_chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=save_chunk)
    stream.start()
    return stream, audio_chunks


# Stop the recording, join all chunks, and save them as a WAV file.
def stop_recording(stream, audio_chunks, wav_path):
    stream.stop()
    stream.close()

    if len(audio_chunks) == 0:
        raise ValueError("No audio was recorded.")

    audio = np.concatenate(audio_chunks)
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path


# Convert any audio file (mp3, m4a, wav...) into a 16 kHz mono WAV file.
def convert_to_wav(input_path, wav_path):
    from faster_whisper import decode_audio  # uses the ffmpeg built into the PyAV library

    audio = decode_audio(input_path, sampling_rate=SAMPLE_RATE)
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path


# Return the length of a WAV file in seconds.
def get_duration(wav_path):
    return sf.info(wav_path).duration


# Quick test: record 5 seconds and save to test_recording.wav
if __name__ == "__main__":
    import time

    if not microphone_available():
        print("No microphone found!")
    else:
        print("Recording for 5 seconds... speak now")
        stream, chunks = start_recording()
        time.sleep(5)
        stop_recording(stream, chunks, "test_recording.wav")
        print("Saved test_recording.wav")
