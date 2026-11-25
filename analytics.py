# Talk-time analytics: who spoke how long, what share of the talking, and how many times.
# Input is the list of speaker turns from diarization: [{"start": 0.0, "end": 3.2, "speaker": "Rahul"}, ...]


# Turn seconds into "m:ss", e.g. 83.4 -> "1:23"
def format_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# Count turns per speaker. Back-to-back turns of the same person count as one turn.
def count_turns(turns):
    counts = {}
    previous_speaker = None
    for turn in sorted(turns, key=lambda t: t["start"]):
        if turn["speaker"] != previous_speaker:
            counts[turn["speaker"]] = counts.get(turn["speaker"], 0) + 1
            previous_speaker = turn["speaker"]
    return counts


# Total speaking time per speaker in seconds.
def speaking_seconds(turns):
    totals = {}
    for turn in turns:
        totals[turn["speaker"]] = totals.get(turn["speaker"], 0.0) + (turn["end"] - turn["start"])
    return totals


# One row per speaker, the biggest talker first:
# [{"Speaker": "Rahul", "Speaking time": "0:31", "Seconds": 31.2, "Share of talk time (%)": 55.0, "Turns": 7}, ...]
def talk_time_table(turns):
    totals = speaking_seconds(turns)
    turn_counts = count_turns(turns)
    all_talk = sum(totals.values())

    rows = []
    for speaker, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        rows.append({
            "Speaker": speaker,
            "Speaking time": format_duration(seconds),
            "Seconds": round(seconds, 1),
            "Share of talk time (%)": round(100 * seconds / all_talk, 1) if all_talk > 0 else 0.0,
            "Turns": turn_counts.get(speaker, 0),
        })
    return rows


# Quick test with made-up turns
if __name__ == "__main__":
    sample_turns = [
        {"start": 0.0, "end": 10.0, "speaker": "Rahul"},
        {"start": 10.5, "end": 14.0, "speaker": "Priya"},
        {"start": 14.2, "end": 16.0, "speaker": "Priya"},   # same person again -> still one turn
        {"start": 17.0, "end": 22.0, "speaker": "Rahul"},
    ]
    for row in talk_time_table(sample_turns):
        print(row)
