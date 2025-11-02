# 🧠 AI Agent Architecture Document

## System Components
| Component | Description |
|------------|-------------|
| Model | `google/flan-t5-base` |
| Adapter | LoRA fine-tuned adapter |
| Tokenizer | AutoTokenizer |
| UI | Streamlit web interface |

## Interaction Flow
User → Streamlit UI → Tokenizer → LoRA Model → Output → Display
