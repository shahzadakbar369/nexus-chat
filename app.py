import os
from dotenv import load_dotenv
from flask import Flask, request, Response, render_template, stream_with_context, jsonify
from groq import Groq, RateLimitError, AuthenticationError

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = Flask(__name__)

# ---------- CONFIG (all in one place, easy to explain) ----------
MODELS = {
    "openai/gpt-oss-20b": "Fast (20B)",
    "openai/gpt-oss-120b": "Smart (120B)",
    "qwen/qwen3.8-27b": "Qwen (27B)",
}
DEFAULT_MODEL = "openai/gpt-oss-20b"

PERSONAS = {
    "general": "You are NetSol Assistant, a helpful and concise AI assistant.",
    "onboarding": (
        "You are an onboarding buddy for new interns at a software company. "
        "Explain workplace basics (Git, code reviews, standups, ticketing, professional email) "
        "in a friendly, simple way with small examples."
    ),
    "coder": (
        "You are a coding mentor. Explain code step by step in simple language, "
        "show short examples in code blocks, and mention common mistakes."
    ),
    "eli5": "Explain everything as if to a curious 10-year-old. Use analogies. Keep it short.",
}
STYLE_RULES = " Use short paragraphs and Markdown. If unsure, say so honestly. Never invent facts."

MAX_HISTORY_CHARS = 12000   # ~3000 tokens: keeps us safely under the 8K tokens/min free limit
MAX_MSG_CHARS = 4000        # reject giant pasted inputs
# ---------------------------------------------------------------


def trim_history(history):
    """Keep the most recent messages that fit inside MAX_HISTORY_CHARS."""
    kept, total = [], 0
    for msg in reversed(history):              # walk from newest to oldest
        total += len(msg["content"])
        if total > MAX_HISTORY_CHARS:
            break
        kept.append(msg)
    return list(reversed(kept))                # restore the original order


def clean_history(raw):
    """Never trust the browser: keep only valid roles and string content."""
    cleaned = []
    for m in raw if isinstance(raw, list) else []:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            cleaned.append({"role": m["role"], "content": m["content"][:MAX_MSG_CHARS]})
    return cleaned


@app.route("/")
def home():
    return render_template("index.html", models=MODELS, personas=list(PERSONAS))


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    history = trim_history(clean_history(data.get("messages")))
    if not history:
        return jsonify(error="Empty message"), 400

    model = data.get("model") if data.get("model") in MODELS else DEFAULT_MODEL
    persona = PERSONAS.get(data.get("persona"), PERSONAS["general"])
    messages = [{"role": "system", "content": persona + STYLE_RULES}] + history

    # gpt-oss models "think" before answering; low effort = faster, cheaper replies
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=1000,
                temperature=0.6,
                **extra,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except RateLimitError:
            yield "\n\n⚠️ Rate limit reached. Wait a few seconds or switch model."
        except AuthenticationError:
            yield "\n\n⚠️ Invalid API key. Check your .env file."
        except Exception as e:
            yield f"\n\n⚠️ Something went wrong: {e}"

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Receives a recorded audio file from the browser and returns the text (Whisper)."""
    audio = request.files.get("audio")
    if not audio:
        return jsonify(error="No audio received"), 400

    data = audio.read()
    if len(data) < 1000:                       # basically silence / too short
        return jsonify(error="Recording too short"), 400
    if len(data) > 5 * 1024 * 1024:            # 5 MB cap protects our free quota
        return jsonify(error="Recording too long"), 413

    try:
        result = client.audio.transcriptions.create(
            file=(audio.filename or "recording.webm", data),   # (name, bytes) tuple
            model="whisper-large-v3-turbo",                    # fast + cheap Whisper variant
        )
        return jsonify(text=result.text.strip())
    except RateLimitError:
        return jsonify(error="Voice limit reached, try again shortly"), 429
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)