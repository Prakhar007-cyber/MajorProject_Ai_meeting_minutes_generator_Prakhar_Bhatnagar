import os
from groq import Groq

# The answer we ask for when the transcript does not contain the answer
NOT_DISCUSSED = "Not discussed in the meeting."

# Instructions we give to the LLM. The whole transcript is the only source of truth.
QA_PROMPT = """You answer questions about one meeting, using ONLY the transcript below.
Rules:
- Use only facts that are in the transcript. Do not guess and do not use outside knowledge.
- If the transcript does not contain the answer, reply exactly: "{not_discussed}"
- Mention who said it (the speaker name) when it helps.
- Keep the answer short: 1-3 sentences.

Transcript:
{transcript}

Question: {question}
Answer:"""


# Answer a question about the meeting from its transcript. Returns None if there is no API key.
def answer_question(transcript_text, question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_your"):
        return None

    prompt = QA_PROMPT.format(not_discussed=NOT_DISCUSSED, transcript=transcript_text, question=question.strip())
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # same question -> same answer, no creativity needed
    )
    return response.choices[0].message.content.strip()


# Quick test with a tiny fake transcript
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")  # so the Windows terminal can print any character
    load_dotenv()
    sample = (
        "Rahul: Let's launch the website on Friday.\n"
        "Priya: Okay, I will finish the login page by Thursday."
    )
    for q in ["When is the launch?", "Who is doing the login page?", "What is the budget?"]:
        print("Q:", q)
        print("A:", answer_question(sample, q))
