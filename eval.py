#!/usr/bin/env python3
"""eval.py - small ROUGE-based evaluator for structured fields.
Usage:
    python eval.py pred.jsonl gold.jsonl
"""

import sys, json
from rouge_score import rouge_scorer

def score_pair(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    scores = {}
    for k in ['background','methods','results','limitations']:
        p = pred.get('summary',{}).get(k,'')
        g = gold.get('summary',{}).get(k,'')
        if g.strip()=='':
            scores[k] = None
            continue
        s = scorer.score(g, p)
        scores[k] = s['rougeL'].fmeasure
    return scores

def main():
    pred_file, gold_file = sys.argv[1], sys.argv[2]
    preds = [json.loads(l) for l in open(pred_file)]
    golds = [json.loads(l) for l in open(gold_file)]
    all_scores = []
    for p,g in zip(preds,golds):
        all_scores.append(score_pair(p,g))
    # aggregate
    agg = {}
    for k in ['background','methods','results','limitations']:
        vals = [s[k] for s in all_scores if s[k] is not None]
        agg[k] = sum(vals)/len(vals) if vals else None
    print('Aggregate ROUGE-L F1 per field:')
    for k,v in agg.items():
        print(f'{k}: {v}')
if __name__=='__main__':
    main()
