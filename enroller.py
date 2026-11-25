import os
import re
import numpy as np
import soundfile as sf
import torch

# Folder where each enrolled person's voice sample is kept as <name>.wav
ENROLLED_FOLDER = "enrolled"

# How similar two voices must be (cosine similarity, -1 to 1) to count as the same person.
# Raise it if wrong names appear, lower it if real people are not recognised.
MATCH_THRESHOLD = 0.5

# A voice sample shorter than this is too short to describe a voice well
MIN_SAMPLE_SECONDS = 3

# Speaker turns shorter than this are skipped (too short to describe a voice)
MIN_TURN_SECONDS = 0.5

# At most this many seconds of each meeting speaker are used to make their voice print
MAX_SPEAKER_SECONDS = 30

# The same speaker-embedding model that pyannote 3.1 uses inside diarization (already downloaded)
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

os.makedirs(ENROLLED_FOLDER, exist_ok=True)


# Load the speaker-embedding model. It turns any piece of speech into a 256-number "voice print".
def load_embedding_model(hf_token):
    from pyannote.audio import Model, Inference

    model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=hf_token)
    if model is None:
        raise ValueError("Could not load the speaker-embedding model. Check HF_TOKEN.")
    return Inference(model, window="whole")  # "whole" = one voice print for the whole clip


# Read a WAV file as mono float32 audio. Returns (audio, sample_rate).
def read_wav_mono(wav_path):
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono
    return audio, sample_rate


# Turn a piece of audio into a voice print (a list of 256 numbers).
def get_embedding(model, audio, sample_rate):
    waveform = torch.from_numpy(audio).unsqueeze(0)  # shape: (1 channel, samples)
    embedding = model({"waveform": waveform, "sample_rate": sample_rate})
    return np.asarray(embedding).reshape(-1)


# Cosine similarity: 1 = same direction (same voice), 0 = unrelated, -1 = opposite.
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Make a name safe to use as a file name: keep letters, digits, spaces, "-" and "_".
def clean_name(name):
    return re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()


# Save a voice sample (any 16 kHz WAV) for a person. Returns the saved path.
def save_enrolled_sample(name, wav_path):
    name = clean_name(name)
    if not name:
        raise ValueError("Please type a name using letters or digits.")

    audio, sample_rate = read_wav_mono(wav_path)
    seconds = len(audio) / sample_rate
    if seconds < MIN_SAMPLE_SECONDS:
        raise ValueError(f"The sample is only {seconds:.1f} s long. Please record at least {MIN_SAMPLE_SECONDS} seconds.")

    saved_path = os.path.join(ENROLLED_FOLDER, name + ".wav")
    sf.write(saved_path, audio, sample_rate)
    return saved_path


# Return all enrolled people as {name: wav_path}, sorted by name.
def list_enrolled():
    people = {}
    for file_name in sorted(os.listdir(ENROLLED_FOLDER)):
        if file_name.lower().endswith(".wav"):
            people[file_name[:-4]] = os.path.join(ENROLLED_FOLDER, file_name)
    return people


# Delete one enrolled person.
def delete_enrolled(name):
    path = os.path.join(ENROLLED_FOLDER, clean_name(name) + ".wav")
    if os.path.exists(path):
        os.remove(path)


# Join all the turns of one meeting speaker into one clip (skipping very short turns).
def speaker_audio(audio, sample_rate, turns, speaker):
    pieces = []
    total_seconds = 0.0
    for turn in turns:
        length = turn["end"] - turn["start"]
        if turn["speaker"] != speaker or length < MIN_TURN_SECONDS:
            continue
        start = int(turn["start"] * sample_rate)
        end = int(turn["end"] * sample_rate)
        pieces.append(audio[start:end])
        total_seconds += length
        if total_seconds >= MAX_SPEAKER_SECONDS:
            break
    if len(pieces) == 0:
        return None
    return np.concatenate(pieces)


# Compare every meeting speaker with every enrolled person.
# Returns (names, scores):
#   names  = {"Person 1": "Rahul", ...} only for matches above MATCH_THRESHOLD
#   scores = [("Person 1", "Rahul", 0.83), ...] best similarity of each speaker, for display
def match_speakers(wav_path, turns, model):
    enrolled = list_enrolled()
    if len(enrolled) == 0:
        return {}, []

    # Voice print of every enrolled person
    enrolled_prints = {}
    for name, path in enrolled.items():
        audio, sample_rate = read_wav_mono(path)
        enrolled_prints[name] = get_embedding(model, audio, sample_rate)

    # Voice print of every speaker found in the meeting
    audio, sample_rate = read_wav_mono(wav_path)
    speaker_prints = {}
    for speaker in sorted(set(turn["speaker"] for turn in turns)):
        clip = speaker_audio(audio, sample_rate, turns, speaker)
        if clip is not None:
            speaker_prints[speaker] = get_embedding(model, clip, sample_rate)

    # Every (similarity, speaker, name) pair, best first
    pairs = []
    for speaker, speaker_print in speaker_prints.items():
        for name, name_print in enrolled_prints.items():
            pairs.append((cosine_similarity(speaker_print, name_print), speaker, name))
    pairs.sort(reverse=True)

    # Give names greedily: best pair first, each speaker and each name used only once
    names = {}
    for similarity, speaker, name in pairs:
        if similarity < MATCH_THRESHOLD:
            break
        if speaker not in names and name not in names.values():
            names[speaker] = name

    # Best score of each speaker (shown in the UI to help tune the threshold)
    scores = []
    for speaker in speaker_prints:
        similarity, _, name = max(p for p in pairs if p[1] == speaker)
        scores.append((speaker, name, round(similarity, 2)))
    return names, scores


# Replace "Person N" with the matched name in every turn. Unmatched speakers keep "Person N".
def apply_names(turns, names):
    for turn in turns:
        turn["speaker"] = names.get(turn["speaker"], turn["speaker"])
    return turns


# Quick test: python enroller.py meeting.wav   (uses the people in the enrolled/ folder)
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from diarizer import load_diarization_pipeline, diarize, rename_speakers

    load_dotenv()
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "test_recording.wav"
    hf_token = os.getenv("HF_TOKEN")

    turns = rename_speakers(diarize(wav_file, load_diarization_pipeline(hf_token)))
    names, scores = match_speakers(wav_file, turns, load_embedding_model(hf_token))
    print("Enrolled:", list(list_enrolled().keys()))
    for speaker, name, similarity in scores:
        print(f"{speaker}: best match {name} (similarity {similarity}) -> {names.get(speaker, speaker)}")
