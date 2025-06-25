#!/usr/bin/env python3
"""
Multilingual Coding Chatbot using OpenRouter API only
"""

from flask import Flask, request, Response, send_file, jsonify, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__, template_folder='.')
CORS(app)

# Use Render's env variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Serve the index.html from current directory
@app.route("/")
def index():
    return send_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"response": "❌ Error: Empty message received."})

    def generate():
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                stream=True,
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

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.get("content"):
                    yield delta["content"]
        except Exception as e:
            yield f"❌ Error: {str(e)}"

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
