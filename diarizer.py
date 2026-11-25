import os
import soundfile as sf
import torch
from pyannote.audio import Pipeline

# pyannote looks at the audio through a 10-second window. By default the window moves 1 second
# at a time (very slow on CPU). Moving it 2.5 seconds at a time is about 2.4x faster.
WINDOW_STEP_SECONDS = 2.5


# Load the pyannote speaker diarization model (needs a Hugging Face token the first time).
def load_diarization_pipeline(hf_token):
    torch.set_num_threads(os.cpu_count())  # use all CPU cores
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    # pyannote returns None (instead of an error) when the token is wrong or terms are not accepted
    if pipeline is None:
        raise ValueError("Could not load pyannote. Check HF_TOKEN and accept the model terms (see README).")
    pipeline._segmentation.step = WINDOW_STEP_SECONDS
    return pipeline


# Find who spoke when. Returns [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"}, ...]
def diarize(wav_path, pipeline, min_speakers=2, max_speakers=5):
    # Read the audio ourselves, so pyannote does not need ffmpeg
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono
    waveform = torch.from_numpy(audio).unsqueeze(0)  # shape: (1 channel, samples)

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


# Rename "SPEAKER_00", "SPEAKER_01" to "Person 1", "Person 2" in order of who spoke first.
def rename_speakers(turns):
    new_names = {}
    for turn in turns:
        if turn["speaker"] not in new_names:
            new_names[turn["speaker"]] = f"Person {len(new_names) + 1}"
        turn["speaker"] = new_names[turn["speaker"]]
    return turns


# Pick the speaker whose turn overlaps the most with this Whisper segment.
def find_speaker(segment, turns):
    best_speaker = "Unknown"
    best_overlap = 0.0
    for turn in turns:
        overlap = min(segment["end"], turn["end"]) - max(segment["start"], turn["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    # No overlap at all: use the turn whose middle is closest to the segment's middle
    if best_speaker == "Unknown" and len(turns) > 0:
        segment_middle = (segment["start"] + segment["end"]) / 2
        closest = min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - segment_middle))
        best_speaker = closest["speaker"]
    return best_speaker


# Give each Whisper segment a speaker, and join back-to-back lines from the same person.
def merge_transcript_with_speakers(segments, turns):
    lines = []
    for segment in segments:
        speaker = find_speaker(segment, turns)
        if len(lines) > 0 and lines[-1]["speaker"] == speaker:
            lines[-1]["text"] += " " + segment["text"]
        else:
            lines.append({"speaker": speaker, "text": segment["text"]})
    return lines


# Turn the list of lines into plain text: "Person 1: Hello"
def format_transcript(lines):
    return "\n\n".join(f"{line['speaker']}: {line['text']}" for line in lines)


# Quick test: python diarizer.py some_meeting.wav
if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    from transcriber import load_whisper_model, transcribe

    load_dotenv()
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "test_recording.wav"

    speaker_turns = rename_speakers(diarize(wav_file, load_diarization_pipeline(os.getenv("HF_TOKEN"))))
    for t in speaker_turns:
        print(f"[{t['start']:.1f}s - {t['end']:.1f}s] {t['speaker']}")

    whisper_segments = transcribe(wav_file, load_whisper_model())
    print("\n" + format_transcript(merge_transcript_with_speakers(whisper_segments, speaker_turns)))
