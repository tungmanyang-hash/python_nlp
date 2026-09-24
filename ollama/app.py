import os
import json

from flask import Flask, request, Response, render_template
from ollama import Client


app = Flask(__name__)

OLLAMA_HOST = "http://localhost:11434"

MODEL_NAME = "gemma4:e2b" # gpt-oss:20b # qwen3.5:0.8b # gemma4:e2b
HISTORY_FILE = "chat_history.json"


client = Client(
    host=OLLAMA_HOST,
    timeout=600
)

def load_history():
    if not os.path.exits(HISTORY_FILE):
        return []
    
    with open(HISTORY_FILE,"r", encoding='utf-8') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)