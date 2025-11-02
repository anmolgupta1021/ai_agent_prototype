# AI Agent Prototype - Research Note Assistant (Prototype)

This archive contains a runnable prototype codebase for training a LoRA adapter and a simple agent skeleton.
Change the MODEL variable in `train_lora.py` to your desired base model before training.

## Contents
- train_lora.py          : LoRA training script (transformers + peft)
- agent_executor.py      : Inference loader that loads base model + LoRA adapter
- ingest.py              : Simple PDF -> text extractor (PyMuPDF)
- planner.py             : Minimal planner that produces subtasks (uses agent_executor)
- eval.py                : Small evaluation helper using rouge
- requirements.txt       : Suggested Python packages
- sample_data/train.jsonl: 2 toy training examples
- sample_data/val.jsonl  : 1 toy validation example
- README.md              : this file

## Notes
- The scripts assume you have access to an appropriate Hugging Face-compatible base model.
- If you're running on limited hardware, choose a smaller base model (e.g., a 7B or smaller).
- Edit `MODEL` in `train_lora.py` and `BASE_MODEL` in `agent_executor.py` to match the model you want to use.

## How to use
1. Create a Python environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare `sample_data/*.jsonl` or your own dataset in the same format.
3. Run training (example):
   ```bash
   python train_lora.py --train_file sample_data/train.jsonl --validation_file sample_data/val.jsonl --output_dir lora_checkpoint
   ```
4. Run inference:
   ```bash
   python agent_executor.py --adapter lora_checkpoint --input_file sample_data/val.jsonl
   ```

