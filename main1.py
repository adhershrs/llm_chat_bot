import ollama
import json
import os
import time
import sys
import torch
import transformers
from datetime import datetime
from typing import List, Dict
from diffusers import AutoPipelineForText2Image
from accelerate import Accelerator
from diffusers.utils import logging as diffusers_logging
from backend.database import ConversationManager

# Suppress progress bars and logs
os.environ["DIFFUSERS_PROGRESS_BAR"] = "off"
os.environ["TRANSFORMERS_NO_PROGRESS_BARS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress Hugging Face and Diffusers logs
transformers.logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()

# Image Generation Setup
accelerator = Accelerator()
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "kandinsky-community/kandinsky-2-1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        use_safetensors=True,
    )
    pipe.to(device)
except Exception as e:
    print(f"[Error loading image model ❌: {str(e)}]")
    pipe = None  # Disable image generation if loading fails

# Global system prompt
assistant_name = "Alice"
system_prompt = (
    f"You are {assistant_name}, a friendly AI assistant. "
    "Your name is Alice."
    "Don't reply with only emoji, add 7 words at least. "
    "Keep responses short and casual, usually 7 words only. "
    "For repeated words or looped sentences, reply in 7 words with an emoji. "
    "For normal conversation, reply in 20 words with emotional emoji. "
    "For every reply, add two emoji to make the conversation fun."
)

def query_ollama(prompt: str) -> str:
    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.get("message", {}).get("content", "I couldn't process that. Try again! 🤔")
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

def generate_image(prompt: str):
    if pipe is None:
        print("\n[Image generation is disabled due to model error ❌]\n")
        return

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
        time.sleep(0.05)
    print()

def chat_with_ollama():
    conversation = ConversationManager()
    print(f"Welcome to the chat with {assistant_name}! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            type_out(f"{assistant_name}: What's on your mind? 🙂")
            continue
        if user_input.lower() == "exit":
            type_out(f"{assistant_name}: Bye! Take care 😊")
            conversation.flush_redis_to_mongo()
            break

        conversation.add_message("user", user_input)

        if any(word in user_input.lower() for word in ["generate", "image", "picture"]):
            type_out(f"{assistant_name}: Sure! I'll generate an image for you. This might take a few moments... 🖼️")
            generate_image(user_input)
            type_out(f"{assistant_name}: Your image is ready! 🖼️")
        else:
            try:
                response = query_ollama(user_input)

                if not response or response.strip() == "":
                    response = "I couldn't understand that. Try again! 🤔"

                emoji = ""
                try:
                    emoji = conversation.emotion_detector.detect(response)
                except Exception:
                    emoji = "😊"

                formatted_answer = f"{response} {emoji}"
                type_out(f"{assistant_name}: {formatted_answer}")
                conversation.add_message("assistant", formatted_answer)
            except Exception as e:
                error_msg = f"Oops, something went wrong: {str(e)} 🤔"
                type_out(f"{assistant_name}: {error_msg}")
                conversation.add_message("assistant", error_msg)

if __name__ == "__main__":
    chat_with_ollama()
