import pandas as pd
import numpy as np
from patch_scout_app import PatchScoutTool
import math

DATA_FILE = "osrs_master_dataset.csv"

TEST_SET = [
    {"query": "Scythe of Vitur", "required_keywords": ["scythe", "vitur"]},
    {"query": "Forestry woodcutting", "required_keywords": ["forestry", "woodcutting", "events"]},
    {"query": "Fang nerf", "required_keywords": ["fang", "nerf", "damage", "slash"]},
    {"query": "Desert Treasure 2 rewards", "required_keywords": ["desert treasure", "dt2", "virtus", "soulreaper"]},
    {"query": "mobile tile markers", "required_keywords": ["mobile", "tile", "marker", "ground"]},
    {"query": "wildy boss rework", "required_keywords": ["wildy", "wilderness", "boss", "artio", "calvar", "spindel"]},
    {"query": "runecrafting outfit", "required_keywords": ["runecraft", "outfit", "raiments", "gotr"]},
    {"query": "blowpipe nerf", "required_keywords": ["blowpipe", "nerf", "dart", "damage"]},
    {"query": "tumeken shadow", "required_keywords": ["tumeken", "shadow", "magic", "staff"]},
    {"query": "quest speedrunning", "required_keywords": ["speedrun", "quest", "timer"]}
]

def calculate_dcg(relevances, k):
    ## Discounted Cumulative Gain
    dcg = 0.0
    for i in range(min(len(relevances), k)):
        rel = relevances[i]
        dcg += rel / math.log2(i + 2) # rank starts at 1, so log2(i+2)
    return dcg

def calculate_ndcg(relevances, k):
    ## Normalized DCG
    dcg = calculate_dcg(relevances, k)
    
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    
    if idcg == 0: return 0.0
    return dcg / idcg

def calculate_ap(relevances):
    ## Average Precision
    if sum(relevances) == 0: return 0.0
    
    hits = 0
    sum_precisions = 0
    for i, rel in enumerate(relevances):
        if rel == 1:
            hits += 1
            sum_precisions += hits / (i + 1)
            
    return sum_precisions / hits

def main():
    print("Loading PatchScout for Evaluation...")
    try:
        app = PatchScoutTool(DATA_FILE)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure 'osrs_master_dataset.csv' exists!")
        return

    print(f"\n{'='*60}")
    print(f"RUNNING AUTOMATED EVALUATION (NDCG@10 & MAP)")
    print(f"{'='*60}")
    
    modes = ["keyword", "semantic"]
    results_summary = {mode: {"ndcg": [], "map": []} for mode in modes}

    for mode in modes:
        print(f"\nEvaluating Mode: {mode.upper()}...")
        print(f"{'-'*60}")
        print(f"{'Query':<30} | {'P@5':<8} | {'NDCG@10':<8} | {'AP':<8}")
        print(f"{'-'*60}")

        for test in TEST_SET:
            query = test['query']
            keywords = test['required_keywords']
            
            results = app.search(query, mode=mode, top_k=10)
            
            relevance_vector = []
            for res in results:
                text_lower = res['text'].lower()
                is_relevant = 1 if any(k in text_lower for k in keywords) else 0
                relevance_vector.append(is_relevant)
            
            while len(relevance_vector) < 10:
                relevance_vector.append(0)
            
            p_at_5 = sum(relevance_vector[:5]) / 5.0
            ndcg_10 = calculate_ndcg(relevance_vector, 10)
            avg_prec = calculate_ap(relevance_vector)
            
            results_summary[mode]["ndcg"].append(ndcg_10)
            results_summary[mode]["map"].append(avg_prec)
            
            print(f"{query[:28]:<30} | {p_at_5:.2f}     | {ndcg_10:.2f}     | {avg_prec:.2f}")

    # FINAL REPORT
    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    for mode in modes:
        avg_ndcg = np.mean(results_summary[mode]["ndcg"])
        mean_ap = np.mean(results_summary[mode]["map"])
        print(f"MODE: {mode.upper()}")
        print(f"  > Mean Average Precision (MAP): {mean_ap:.4f}")
        print(f"  > Average NDCG@10:              {avg_ndcg:.4f}")
        print("")

if __name__ == "__main__":
    main()