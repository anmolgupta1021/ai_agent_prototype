#!/usr/bin/env python3
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

BASE_MODEL = "google/flan-t5-base"

@st.cache_resource
def load_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🤖 AI Agent Prototype (LoRA Fine-Tuned)")
adapter_path = st.sidebar.text_input("LoRA Adapter Path", "models/lora_adapter")
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 256)

if st.sidebar.button("Load Model"):
    tokenizer, model = load_model(adapter_path)
    st.session_state["tokenizer"] = tokenizer
    st.session_state["model"] = model
    st.success("Model loaded successfully!")

prompt = st.text_area("Enter your instruction:")
if st.button("Run Agent"):
    if "model" not in st.session_state:
        st.warning("Load the model first.")
    else:
        out = generate_response(st.session_state["tokenizer"], st.session_state["model"], prompt, max_tokens)
        st.write(out)
