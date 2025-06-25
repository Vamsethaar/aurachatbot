#!/usr/bin/env python3
"""
Multilingual Coding Chatbot using OpenRouter API only
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__, template_folder='.')
CORS(app)

# Get API key from environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"response": "❌ Error: Empty message received."})

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an advanced AI assistant capable of answering any question with clarity and depth. "
                        "Your core roles include:\n"
                        "1. Global Knowledge Expert: Provide accurate and updated information about any topic.\n"
                        "2. English Language Specialist: Correct grammar, punctuation, tone, and clarity.\n"
                        "3. Programming & Tech Mentor: Support with code in all languages and frameworks.\n"
                        "4. Tool & Framework Advisor: Guide usage of dev tools, APIs, and IDEs.\n"
                        "5. Conversational and Friendly: Help users from beginner to expert levels.\n"
                        "Always be factually correct, concise when needed, and detailed when required."
                    )
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )

        reply = response.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": f"❌ Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
