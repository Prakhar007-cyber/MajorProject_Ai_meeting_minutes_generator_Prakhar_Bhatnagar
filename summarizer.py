import os
import re
import json
from groq import Groq

# Instructions we give to the LLM. Action items come back as JSON so we can show them as a table.
PROMPT = """You are a meeting assistant. Read the meeting transcript below and write short meeting minutes.
Use exactly these three headings:

Summary:
(3-5 sentences)

Key Decisions:
- (one bullet per decision, or "- None")

Action Items:
(only a JSON list, one object per task, using the speaker names from the transcript, like this:
[{{"person": "Name", "task": "what they will do", "deadline": "when, or Not specified"}}]
Write [] if there are no action items.)

Transcript:
{transcript}
"""


# Send the transcript to the Groq LLM and get back meeting minutes as text.
# Returns None if there is no API key. To use another LLM, only change this function.
def generate_minutes(transcript_text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_your"):
        return None

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": PROMPT.format(transcript=transcript_text)}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# Split the minutes into (text before "Action Items:", list of action items).
# Each item is {"Person": ..., "Task": ..., "Deadline": ...}.
# If the action items cannot be read, returns (the full minutes, None) so the old text display is used.
def split_action_items(minutes_text):
    heading = re.search(r"action items", minutes_text, re.IGNORECASE)  # works with "**Action Items:**" too
    if heading is None:
        return minutes_text, None
    position = heading.start()

    found = re.search(r"\[.*\]", minutes_text[position:], re.DOTALL)  # the JSON list after the heading
    if found is None:
        return minutes_text, None
    try:
        raw_items = json.loads(found.group(0))
        items = []
        for item in raw_items:
            items.append({
                "Person": str(item.get("person", "")).strip() or "Not specified",
                "Task": str(item["task"]).strip(),
                "Deadline": str(item.get("deadline", "")).strip() or "Not specified",
            })
    except (ValueError, KeyError, TypeError, AttributeError):
        return minutes_text, None
    return minutes_text[:position].rstrip("*# \n"), items  # also drop a "**" or "##" left before the heading


# Turn action items back into plain text lines (for the downloaded .txt file).
def format_action_items(items):
    if len(items) == 0:
        return "Action Items:\n- None"
    lines = [f"- {item['Person']} - {item['Task']} (deadline: {item['Deadline']})" for item in items]
    return "Action Items:\n" + "\n".join(lines)


# Quick test with a tiny fake transcript
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")  # so the Windows terminal can print any character
    load_dotenv()
    sample = (
        "Person 1: Let's launch the website on Friday.\n"
        "Person 2: Okay, I will finish the login page by Thursday.\n"
        "Person 1: Great, and Rahul will test it on Thursday evening."
    )
    minutes = generate_minutes(sample)
    print(minutes if minutes else "No GROQ_API_KEY found in .env, minutes skipped.")

    if minutes:
        text, items = split_action_items(minutes)
        print("\n--- Parsed action items ---")
        print(items if items is not None else "Could not parse, the app will show the text above.")

    # The parser must never crash on bad LLM output
    print(split_action_items("Action Items:\n- Rahul - test the site"))      # no JSON -> (text, None)
    print(split_action_items('Action Items:\n[{"person": "Rahul"'))         # broken JSON -> (text, None)
