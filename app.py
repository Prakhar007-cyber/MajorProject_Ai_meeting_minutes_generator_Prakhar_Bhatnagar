import os
import time
from dotenv import load_dotenv

load_dotenv()  # read keys from the .env file before anything else

import pandas as pd
import streamlit as st
from recorder import microphone_available, start_recording, stop_recording, convert_to_wav, get_duration
from transcriber import load_whisper_model, transcribe
from diarizer import load_diarization_pipeline, diarize, rename_speakers, merge_transcript_with_speakers, format_transcript
from summarizer import generate_minutes, split_action_items, format_action_items
from qa import answer_question
from analytics import talk_time_table
from enroller import load_embedding_model, match_speakers, apply_names, list_enrolled, save_enrolled_sample, delete_enrolled, MATCH_THRESHOLD

RECORDINGS_FOLDER = "recordings"
LANGUAGES = {"English": "en", "Hindi": "hi", "Auto-detect": None}
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)


# Load Whisper only once (Streamlit keeps it in memory between button clicks)
@st.cache_resource
def get_whisper_model():
    return load_whisper_model("small")


# Load pyannote only once
@st.cache_resource
def get_diarization_pipeline(hf_token):
    return load_diarization_pipeline(hf_token)


# Load the speaker-embedding model (for enrolled names) only once
@st.cache_resource
def get_embedding_model(hf_token):
    return load_embedding_model(hf_token)


# Return the Hugging Face token from .env, or None if it is missing.
def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if not token or token.startswith("hf_your"):
        return None
    return token


# Add speaker names to the Whisper segments. If diarization fails, use "Speaker" for every line.
def add_speakers(wav_path, segments, min_speakers, max_speakers):
    hf_token = get_hf_token()
    if hf_token is None:
        st.warning("HF_TOKEN missing in .env, so speakers are not labeled. See README.")
        return [{"speaker": "Speaker", "text": s["text"]} for s in segments]

    try:
        with st.spinner("Step 2/3: Finding who spoke when (pyannote)..."):
            pipeline = get_diarization_pipeline(hf_token)
            turns = rename_speakers(diarize(wav_path, pipeline, min_speakers, max_speakers))
    except Exception as error:
        st.warning(f"Speaker labeling failed ({error}). Check your HF token and model terms. Showing transcript without speakers.")
        return [{"speaker": "Speaker", "text": s["text"]} for s in segments]

    turns = add_enrolled_names(wav_path, turns, hf_token)
    st.session_state.turns = turns
    return merge_transcript_with_speakers(segments, turns)


# Replace "Person N" with enrolled names when a voice matches. On any problem, keep "Person N".
def add_enrolled_names(wav_path, turns, hf_token):
    if len(list_enrolled()) == 0:
        return turns
    try:
        with st.spinner("Matching voices with enrolled speakers..."):
            names, scores = match_speakers(wav_path, turns, get_embedding_model(hf_token))
        st.session_state.speaker_matches = scores
        return apply_names(turns, names)
    except Exception as error:
        st.warning(f"Could not match enrolled speakers ({error}). Showing Person 1, 2, ... instead.")
        return turns


# Full pipeline: WAV -> transcript with speakers -> minutes. Saves results in session_state.
def process_audio(wav_path, min_speakers, max_speakers, language):
    st.session_state.transcript = None
    st.session_state.minutes = None
    st.session_state.turns = None
    st.session_state.speaker_matches = None
    st.session_state.qa_history = []
    start_time = time.time()

    audio_seconds = get_duration(wav_path)
    if audio_seconds < 1:
        st.error("The audio is too short or empty. Please record for at least a few seconds.")
        return
    st.caption(f"Audio length: {audio_seconds / 60:.1f} min. On a laptop CPU this takes about the same time to process.")

    with st.spinner("Step 1/3: Converting speech to text (Whisper)..."):
        segments = transcribe(wav_path, get_whisper_model(), language)

    if len(segments) == 0:
        st.error("No speech was detected in the audio. Try speaking louder or closer to the mic.")
        return

    lines = add_speakers(wav_path, segments, min_speakers, max_speakers)
    st.session_state.transcript = format_transcript(lines)

    try:
        with st.spinner("Step 3/3: Writing meeting minutes (LLM)..."):
            st.session_state.minutes = generate_minutes(st.session_state.transcript)
        if st.session_state.minutes is None:
            st.info("GROQ_API_KEY missing in .env, so minutes were skipped. The transcript is below.")
    except Exception as error:
        st.warning(f"Could not generate minutes ({error}). The transcript is still shown below.")

    st.success(f"Done in {time.time() - start_time:.0f} seconds.")


# ---------------- Page layout ----------------
st.set_page_config(page_title="AI Meeting Minutes Generator", page_icon="📝")
st.title("📝 AI Meeting Minutes Generator")
st.write("Record a meeting, get a speaker-labeled transcript and short minutes.")

# Remember values between button clicks
for key in ["stream", "chunks", "transcript", "minutes", "turns", "speaker_matches", "enroll_stream", "enroll_chunks"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

min_speakers, max_speakers = st.slider("Number of people in the meeting", 2, 5, (2, 5))
language = LANGUAGES[st.selectbox("Meeting language", list(LANGUAGES.keys()))]

# --- Optional: enroll speakers so the transcript shows real names ---
with st.expander("👤 Enroll speakers (optional): show names instead of Person 1, 2, ..."):
    st.write("Type a name, then record 5-10 seconds of that person speaking, or upload a WAV of their voice.")
    enroll_name = st.text_input("Name", placeholder="e.g. Rahul")
    enroll_path = os.path.join(RECORDINGS_FOLDER, "enroll_sample.wav")
    rec_col, stop_col_e = st.columns(2)

    if rec_col.button("🎙️ Record sample", disabled=st.session_state.enroll_stream is not None, use_container_width=True):
        if not enroll_name.strip():
            st.error("Please type a name first.")
        elif not microphone_available():
            st.error("No microphone found. Upload a WAV sample instead.")
        else:
            try:
                st.session_state.enroll_stream, st.session_state.enroll_chunks = start_recording()
                st.rerun()
            except Exception as error:
                st.error(f"Could not start the microphone: {error}")

    if stop_col_e.button("💾 Stop and save", disabled=st.session_state.enroll_stream is None, use_container_width=True):
        try:
            stop_recording(st.session_state.enroll_stream, st.session_state.enroll_chunks, enroll_path)
            save_enrolled_sample(enroll_name, enroll_path)
            st.success(f"Saved voice sample for {enroll_name.strip()}.")
        except Exception as error:
            st.error(f"Could not save the sample: {error}")
        st.session_state.enroll_stream = None

    if st.session_state.enroll_stream is not None:
        st.info("🔴 Recording the voice sample... ask the person to speak for 5-10 seconds, then click Stop and save.")

    sample_file = st.file_uploader("Or upload a voice sample", type=["wav", "mp3", "m4a", "ogg", "flac"], key="enroll_upload")
    if sample_file is not None and st.button("Save uploaded sample"):
        upload_path = os.path.join(RECORDINGS_FOLDER, "enroll_upload_" + sample_file.name)
        with open(upload_path, "wb") as f:
            f.write(sample_file.getbuffer())
        try:
            save_enrolled_sample(enroll_name, convert_to_wav(upload_path, enroll_path))
            st.success(f"Saved voice sample for {enroll_name.strip()}.")
        except Exception as error:
            st.error(f"Could not save the sample: {error}")

    enrolled_people = list_enrolled()
    if enrolled_people:
        st.write("**Enrolled speakers:**")
        for person, person_path in enrolled_people.items():
            name_col, play_col, remove_col = st.columns([2, 4, 1])
            name_col.write(person)
            play_col.audio(person_path)
            if remove_col.button("Remove", key="remove_" + person):
                delete_enrolled(person)
                st.rerun()
    else:
        st.caption("Nobody is enrolled, so speakers will be shown as Person 1, Person 2, ...")

# --- Option 1: record from the microphone ---
st.subheader("1. Record from microphone")
start_col, stop_col = st.columns(2)

if start_col.button("▶️ Start", disabled=st.session_state.stream is not None, use_container_width=True):
    if not microphone_available():
        st.error("No microphone found. Please connect a mic, or use 'Load existing audio file' below.")
    else:
        try:
            st.session_state.stream, st.session_state.chunks = start_recording()
            st.rerun()
        except Exception as error:
            st.error(f"Could not start the microphone: {error}")

if stop_col.button("⏹️ Stop", disabled=st.session_state.stream is None, use_container_width=True):
    wav_path = os.path.join(RECORDINGS_FOLDER, "meeting.wav")
    try:
        stop_recording(st.session_state.stream, st.session_state.chunks, wav_path)
        st.session_state.stream = None
        st.audio(wav_path)
        process_audio(wav_path, min_speakers, max_speakers, language)
    except ValueError as error:
        st.session_state.stream = None
        st.error(f"{error} Please try again.")

if st.session_state.stream is not None:
    st.info("🔴 Recording... click Stop when the meeting is over.")

# --- Option 2: load an audio file (backup for the demo) ---
st.subheader("2. Or load an existing audio file")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None and st.button("Process this file"):
    upload_path = os.path.join(RECORDINGS_FOLDER, "uploaded_" + uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        wav_path = convert_to_wav(upload_path, os.path.join(RECORDINGS_FOLDER, "uploaded.wav"))
        st.audio(wav_path)
        process_audio(wav_path, min_speakers, max_speakers, language)
    except Exception as error:
        st.error(f"Could not read this audio file: {error}")

# --- Results ---
if st.session_state.transcript:
    st.subheader("📄 Transcript")
    st.markdown(st.session_state.transcript)
    if st.session_state.speaker_matches:
        matches = ", ".join(f"{speaker} → best match {name} ({score})" for speaker, name, score in st.session_state.speaker_matches)
        st.caption(f"Voice matching (threshold {MATCH_THRESHOLD}): {matches}")

    minutes_for_file = st.session_state.minutes
    if st.session_state.minutes:
        st.subheader("🧾 Meeting Minutes")
        minutes_text, action_items = split_action_items(st.session_state.minutes)
        if action_items is None:
            st.markdown(st.session_state.minutes)  # could not read the action items: show the text as before
        else:
            st.markdown(minutes_text)
            st.markdown("**Action Items:**")
            if action_items:
                st.dataframe(pd.DataFrame(action_items), hide_index=True, use_container_width=True)
            else:
                st.write("None")
            minutes_for_file = minutes_text + "\n\n" + format_action_items(action_items)

    # --- Talk-time analytics from the speaker turns ---
    st.subheader("📊 Meeting analytics")
    if st.session_state.turns:
        try:
            table = pd.DataFrame(talk_time_table(st.session_state.turns))
            st.bar_chart(table, x="Speaker", y="Share of talk time (%)", horizontal=True, height=220)
            st.dataframe(table.drop(columns="Seconds"), hide_index=True, use_container_width=True)
        except Exception as error:
            st.warning(f"Could not calculate analytics ({error}).")
    else:
        st.caption("Analytics need speaker labels, which are not available for this meeting (check HF_TOKEN).")

    download_text = "MEETING TRANSCRIPT\n==================\n\n" + st.session_state.transcript
    if minutes_for_file:
        download_text += "\n\n\nMEETING MINUTES\n===============\n\n" + minutes_for_file

    st.download_button("⬇️ Download transcript + minutes (.txt)", download_text, file_name="meeting_minutes.txt")

    # --- Ask the meeting: questions answered only from the transcript ---
    st.subheader("💬 Ask the meeting")
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Ask something about this meeting", placeholder="e.g. Who will add the schedule page?")
        asked = st.form_submit_button("Ask")

    if asked and question.strip():
        try:
            with st.spinner("Reading the transcript..."):
                answer = answer_question(st.session_state.transcript, question)
            if answer is None:
                st.info("GROQ_API_KEY missing in .env, so questions cannot be answered.")
            else:
                st.session_state.qa_history.append((question.strip(), answer))
        except Exception as error:
            st.warning(f"Could not answer the question ({error}). Please try again.")

    for past_question, past_answer in reversed(st.session_state.qa_history):
        st.markdown(f"**Q:** {past_question}  \n**A:** {past_answer}")
