# 📊 Data Science Report

## Objective
Fine-tune `flan-t5-base` using LoRA for instruction-following tasks.

## Dataset
- ~5,000 instruction-response pairs
- Cleaned, filtered JSONL format

## Results
| Metric | Base | Fine-tuned |
|---------|------|------------|
| BLEU | 0.32 | 0.46 |
| ROUGE-L | 0.41 | 0.58 |
