#!/usr/bin/env python3
"""Streamlit app for AI Agent Prototype"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Base model — small & open
BASE_MODEL = "google/flan-t5-base"  # ~250 MB, fits on CPU easily


@st.cache_resource
def load_model(adapter_path):
    """Load LoRA adapter and base model."""
    try:
        st.info(f"Attempting to load base model: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float32,  # use float32 for CPU safety
        )

        # Optional: attach LoRA adapter if it exists
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            st.success("✅ Loaded LoRA adapter successfully.")
        except Exception:
            st.warning("⚠️ No valid LoRA adapter found, using base model only.")
            model = base_model

        model.eval()
        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        raise


def generate_response(tokenizer, model, prompt, max_tokens=256):
    """Generate text for encoder-decoder models like T5."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Agent Prototype", layout="centered")
st.title("🤖 AI Agent Prototype (LoRA Fine-Tuned)")

# Sidebar
st.sidebar.header("⚙️ Configuration")
adapter_path = st.sidebar.text_input("LoRA Adapter Path", "models/lora_adapter")
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 256)

# Load model button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model... This may take a minute ⏳"):
        tokenizer, model = load_model(adapter_path)
    st.session_state["tokenizer"] = tokenizer
    st.session_state["model"] = model
    st.success("✅ Model loaded successfully!")

# Main text input area
user_prompt = st.text_area("Enter your instruction or query:", height=200)

if st.button("Run Agent"):
    if "model" not in st.session_state:
        st.warning("⚠️ Please load the model first from the sidebar.")
    elif not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt first.")
    else:
        with st.spinner("Generating response..."):
            output_text = generate_response(
                st.session_state["tokenizer"],
                st.session_state["model"],
                user_prompt,
                max_tokens=max_tokens,
            )
        st.subheader("🧠 Model Output")
        st.write(output_text)
