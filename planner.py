#!/usr/bin/env python3
"""planner.py - minimal planner that creates subtasks for the executor.

This is a tiny example showing how you might split a paper into subtasks.
"""

import json
from agent_executor import load_model, call_model

def simple_planner(title, abstract=None):
    # Very small deterministic planner
    subtasks = [
        {"name":"extract_background","prompt":"Instruction: Read the abstract and produce a short background paragraph."},
        {"name":"extract_claim","prompt":"Instruction: Read the abstract and conclusion, and state the main claim in one sentence."},
        {"name":"extract_methods","prompt":"Instruction: Read the methods section and produce a concise reproducible summary (3-6 steps)."},
        {"name":"extract_actions","prompt":"Instruction: Based on the paper, propose 3 concrete follow-up experiments or actions."}
    ]
    return subtasks

if __name__ == '__main__':
    import sys
    # demo: load model and run planner+executor on a sample short abstract
    tokenizer, model = load_model('lora_checkpoint')  # change path as needed
    title = 'Sample Paper'
    abstract = 'We present a method that does X. Our results show improvement over baseline A by 5%.'
    tasks = simple_planner(title, abstract)
    for t in tasks:
        prompt = t['prompt'] + '\n\n' + (abstract or '')
        out = call_model(tokenizer, model, prompt, max_tokens=256)
        print('TASK:', t['name'])
        print(out)
        print('---')
