# 🤖 AI Agent Prototype — Research Note Assistant (LoRA Fine-Tuned)

This project implements a fine-tuned AI Agent that assists in organizing and summarizing research notes.  
It uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of large language models and includes a minimal **Streamlit UI** for interaction.

---

## 🧠 Overview

The **Research Note Assistant** automates part of the manual academic workflow — summarizing, organizing, and rewriting research notes or academic papers in structured form.

The model is fine-tuned using **parameter-efficient tuning (LoRA)** on top of a **Flan-T5 Base** model, enabling strong reasoning and summarization while keeping compute and memory costs low.

---

## 📂 Repository Contents

| File / Folder | Description |
|----------------|-------------|
| `train_lora.py` | Fine-tuning script using PEFT (LoRA) and Hugging Face Transformers |
| `agent_executor.py` | Loads the fine-tuned adapter for inference and interaction |
| `ingest.py` | PDF → text converter using PyMuPDF (for preparing fine-tuning data) |
| `planner.py` | Simple planning module that creates subtasks and invokes the agent |
| `eval.py` | Evaluation helper that uses ROUGE metrics for summarization performance |
| `requirements.txt` | Python dependencies |
| `sample_data/train.jsonl` | Example training data (toy dataset) |
| `sample_data/val.jsonl` | Example validation data |
| `src/streamlit_app.py` | Streamlit app for interactive testing |
| `README.md` | This documentation file |

---

## ⚙️ Model and Architecture

- **Base Model:** `google/flan-t5-base`
- **Fine-tuning Technique:** LoRA (via PEFT)
- **Frameworks:** Hugging Face Transformers, PEFT, PyTorch
- **Interface:** Streamlit
- **Purpose:** Automate note summarization and restructuring of academic text

---

## 🧩 Architecture Summary

**Components:**
1. **Planner:** Breaks down user tasks into subtasks (e.g., “Summarize section,” “Extract keywords”).
2. **Executor:** Executes each subtask using the fine-tuned LoRA model.
3. **Evaluator:** Uses ROUGE to assess summary quality.
4. **UI Layer:** Streamlit-based simple front-end for interaction.

**Flow:**
User → Planner → Executor (Flan-T5 + LoRA) → Evaluator → Output Summary

---

## 🚀 Setup & Usage

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
---
```


### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Train the LoRA adapter
``` bash
python train_lora.py \
  --train_file sample_data/train.jsonl \
  --validation_file sample_data/val.jsonl \
  --output_dir models/lora_adapter
```
### 4️⃣ Run inference on validation data
``` bash
python agent_executor.py --adapter models/lora_adapter --input_file sample_data/val.jsonl
```

### 5️⃣ Launch the Streamlit app
``` bash
streamlit run src/streamlit_app.py
```

## 🧪 Evaluation

Evaluation is performed using ROUGE-L and qualitative comparison between human-written and AI-generated summaries.
The eval.py script provides a quick scoring utility.

Example:
```bash
python eval.py --ref sample_data/val.jsonl --pred outputs/generated.jsonl
```

## 📊 Deliverables

✅ Source code of prototype

✅ AI Agent Architecture document (architecture_document.pdf)

✅ Data Science report (data_science_report.pdf)

✅ Interaction logs (interaction_logs.txt)

✅ (Optional) Screenshots or demo video


## 🧰 Dependencies

Key libraries:

transformers

peft

torch

streamlit

PyMuPDF

evaluate

rouge-score

Install all via:

pip install -r requirements.txt

## 🧩 Notes

You may change the base model in train_lora.py by modifying the MODEL variable.

The LoRA adapter output directory (models/lora_adapter) should contain:

adapter_config.json

adapter_model.safetensors

For low-resource machines, use smaller models like google/flan-t5-small.

## 🏁 Example Command Summary
# Fine-tune
python train_lora.py --train_file sample_data/train.jsonl --validation_file sample_data/val.jsonl --output_dir models/lora_adapter

# Run Inference
python agent_executor.py --adapter models/lora_adapter --input_file sample_data/val.jsonl

# Launch Streamlit Interface
streamlit run src/streamlit_app.py

📧 Contact
## 👨‍🎓 Author Information

Name: Anmol Gupta
University: Indian Institute of Technology (IIT) Kanpur
Department: Department of Statistics and Data Science

This project is submitted as part of the AI Internship Application Task.

 
