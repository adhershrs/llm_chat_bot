import ollama
from textblob import TextBlob
import redis
import pymongo
import json
import os
import time
import sys
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict
from diffusers import AutoPipelineForText2Image
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

# Redis and MongoDB Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "chat_history"
MONGO_COLLECTION = "conversations"

MAX_REDIS_HISTORY = 5  # Redis stores only the last 5 messages
MAX_HISTORY_LENGTH = 20  # Fetch max 20 messages for context
TYPING_SPEED = 0.05  # Simulated typing speed (seconds per character)

# Initialize Ollama Model
def query_ollama(prompt: str, history: List[Dict[str, str]]) -> str:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=history + [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# Image Generation Setup
accelerator = Accelerator()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-1", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to(device)

class EmotionDetector:
    def __init__(self):
        self.emotion_thresholds = {
            'very_positive': 0.6,
            'positive': 0.2,
            'neutral': -0.1,
            'negative': -0.2,
            'very_negative': float('-inf')
        }
        self.emotion_emojis = {
            'very_positive': "😊",
            'positive': "🙂",
            'neutral': "😐",
            'negative': "🙁",
            'very_negative': "😔"
        }
    
    def detect(self, text: str) -> str:
        sentiment = TextBlob(text).sentiment
        for emotion, threshold in self.emotion_thresholds.items():
            if sentiment.polarity >= threshold:
                return self.emotion_emojis[emotion]
        return self.emotion_emojis['neutral']

class ConversationManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.mongo_client = pymongo.MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.collection = self.db[MONGO_COLLECTION]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
        self.emotion_detector = EmotionDetector()

    def add_message(self, role: str, content: str):
        """Stores messages along with embeddings for filtering."""
        embedding = self.model.encode(content).tolist()  # Convert to list for JSON storage
        message = {"role": role, "content": content, "embedding": embedding, "timestamp": datetime.now().isoformat()}
        
        self.redis_client.rpush("chat_history", json.dumps(message))

        if self.redis_client.llen("chat_history") > MAX_REDIS_HISTORY:
            old_message = json.loads(self.redis_client.lpop("chat_history"))
            self.collection.insert_one(old_message)

    def get_relevant_context(self, query: str, top_n=5):
        """Retrieves only the most relevant messages instead of full history."""
        query_embedding = self.model.encode(query)

        # Fetch recent messages
        redis_messages = [json.loads(msg) for msg in self.redis_client.lrange("chat_history", -MAX_REDIS_HISTORY, -1)]

        # Calculate similarity and rank messages
        ranked_messages = []
        for msg in redis_messages:
            msg_embedding = np.array(msg["embedding"])
            similarity = np.dot(query_embedding, msg_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(msg_embedding))
            ranked_messages.append((similarity, msg))

        # Sort by similarity (highest first) and select top_n messages
        ranked_messages.sort(reverse=True, key=lambda x: x[0])
        relevant_messages = [msg for _, msg in ranked_messages[:top_n]]

        return [{"role": msg["role"], "content": msg["content"]} for msg in relevant_messages]

    def flush_redis_to_mongo(self):
        """Moves all remaining Redis chat history to MongoDB on session end."""
        while self.redis_client.llen("chat_history") > 0:
            old_message = json.loads(self.redis_client.lpop("chat_history"))
            self.collection.insert_one(old_message)

def generate_image(prompt: str):
    print("\n[Generating image... Please wait 🖼️]\n")
    try:
        image = pipe(
            prompt=prompt, 
            negative_prompt="low quality, bad quality", 
            prior_guidance_scale=1.0, 
            height=512, width=512, 
            num_inference_steps=25
        ).images[0]
        image_path = "generated_image.png"
        image.save(image_path)
        print(f"\n[Image saved as {image_path} ✅]\n")
    except Exception as e:
        print(f"\n[Image generation failed ❌: {str(e)}]\n")

def type_out(text: str):
    """Simulates a typing effect when printing text"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(TYPING_SPEED)
    print()

def chat_with_ollama():
    assistant_name = "Alice"
    system_prompt = (
        f"You are {assistant_name}, a friendly AI assistant. "
        "Your name is Alice."
        "Don't reply with only emoji add 7 words init"
        "Keep responses short and casual, usually 7 words only. "
        "Be natural and concise. Avoid being overly verbose or formal. "
        "for repeade words or looped sentence reply it in 7 word with emoji. "
        "For normal conversation and any repeated conversation reply in 20 words and add emotional emoji. "
        "For every reply add two emoji so it will be fun in conversation."
    )
    conversation = ConversationManager()
    print(f"Welcome to the chat with {assistant_name}! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            type_out(f"{assistant_name}: What's on your mind? 🙂")
            continue
        if user_input.lower() == "exit":
            type_out(f"{assistant_name}: Bye! Take care 😊")
            conversation.flush_redis_to_mongo()  # Move Redis history to MongoDB
            break

        conversation.add_message("user", user_input)

        # Check if user wants an image
        if any(word in user_input.lower() for word in ["generate", "image", "picture"]):
            type_out(f"{assistant_name}: Sure! I'll generate an image for you. This might take a few moments... 🖼️")
            generate_image(user_input)
            type_out(f"{assistant_name}: Your image is ready! 🖼️")
        else:
            try:
                history = conversation.get_relevant_context(user_input, top_n=5)  # Filter messages
                response = query_ollama(user_input, history)
                emoji = conversation.emotion_detector.detect(response)
                formatted_answer = f"{response} {emoji}"
                type_out(f"{assistant_name}: {formatted_answer}")
                conversation.add_message("assistant", formatted_answer)
            except Exception:
                error_msg = "Oops, something went wrong. Let's try again! 🤔"
                type_out(f"{assistant_name}: {error_msg}")
                conversation.add_message("assistant", error_msg)

if __name__ == "__main__":
    chat_with_ollama()
