import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages = [
    {"role": "system", "content": "You are a friendly assistant for NetSol interns. Keep answers short."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})   # 1. remember what user said

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,                                      # 2. send the full history
    )
    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})   # 3. remember what bot said
    print("Bot:", reply)