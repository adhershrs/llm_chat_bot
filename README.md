# LLM Chat Bot

A powerful, locally-run chatbot application that combines state-of-the-art text and image generation models, accelerated by CUDA, with persistent memory capabilities.

## 📝 Description

This project implements an advanced conversational AI assistant capable of both natural language processing and image synthesis. It leverages **Llama 3.2 3B** for generating human-like text responses and integrates the **Kandinsky** model to generate images based on textual prompts. The application is designed to run efficiently on local hardware using **CUDA** acceleration for NVIDIA GPUs.

To enhance the user experience, it uses **Redis** for maintaining immediate conversational context and **MongoDB** for storing long-term conversation history.

## ✨ Features

* **Advanced Conversational AI:** Utilizes Meta's Llama 3.2 3B model for fast and coherent text interactions.
* **Text-to-Image Generation:** Integrated Kandinsky model allows users to create images directly within the chat interface.
* **Context Awareness:** Uses **Redis** to cache and remember the immediate context of the active conversation.
* **Persistent History:** Stores the complete conversation history in **MongoDB** for long-term retrieval.
* **CUDA Accelerated:** Optimized for NVIDIA GPUs to ensure low-latency performance during inference.
* **Local Privacy:** Runs entirely on your local machine, keeping your data private.

## 📂 Project Structure

```bash
llm_chat_bot/
├── backend/              # Backend logic for model, Redis, and MongoDB interaction
├── .gitignore            # Git ignore file
├── generated_image.png   # Sample output from Kandinsky
├── main.py               # Main application entry point
├── main1.py              # [Alternative/Development version]
├── main2.py              # [Alternative/Development version]
└── requirement.txt       # Python dependencies
