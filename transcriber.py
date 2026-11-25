import os
from faster_whisper import WhisperModel


# Load the Whisper model on CPU (using all CPU cores). Try "small" first, and use "base" if that fails.
def load_whisper_model(model_size="small"):
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    except Exception as error:
        print(f"Could not load '{model_size}' model ({error}). Using 'base' instead.")
        return WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())


# Convert a WAV file into a list of segments: [{"start": 0.0, "end": 2.5, "text": "Hello"}, ...]
# language: "en", "hi", ... or None to auto-detect
def transcribe(wav_path, model, language=None):
    segments, info = model.transcribe(
        wav_path,
        language=language,
        vad_filter=True,                   # skip silent parts (Voice Activity Detection)
        beam_size=1,                       # greedy decoding: much faster on CPU
        temperature=0.0,                   # no slow re-tries when the audio is unclear
        condition_on_previous_text=False,  # stops Whisper from repeating itself in loops
    )

    results = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            results.append({"start": segment.start, "end": segment.end, "text": text})
    return results


# Quick test: python transcriber.py some_audio.wav
if __name__ == "__main__":
    import sys

    wav_file = sys.argv[1] if len(sys.argv) > 1 else "test_recording.wav"
    whisper_model = load_whisper_model()
    for seg in transcribe(wav_file, whisper_model):
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
