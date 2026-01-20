#!/usr/bin/env python3
"""Example usage of ranking module."""

from ranking import BradleyTerryRanker, EloRanker, MultiCategoryRanker

# Pairwise comparisons from judge LLM
comparisons_by_category = {
    'temporal_grounding': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
        ('GPT-4V', 'Llama-Vision'),
        ('Claude-3.5', 'Llama-Vision'),
    ],
    'entity_recognition': [
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
        ('GPT-4V', 'Claude-3.5'),
    ],
    'clarification_questions': [
        ('GPT-4V', 'Claude-3.5'),
        ('Gemini', 'Llama-Vision'),
        ('Claude-3.5', 'Gemini'),
    ],
    'answer_quality': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Llama-Vision'),
        ('Claude-3.5', 'Llama-Vision'),
    ]
}

# --- Bradley-Terry Example ---
print("=== Bradley-Terry Rankings ===")
mcr_bt = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)
mcr_bt.fit(comparisons_by_category)

# Export results
mcr_bt.export_csv('bt_rankings.csv')
mcr_bt.export_json('bt_report.json')

# View overall rankings
overall = mcr_bt.get_overall_ranking()
print("\nOverall Rankings:")
for rank, (model, score) in enumerate(overall, 1):
    print(f"{rank}. {model}: {score:.3f}")

# View category-specific rankings
print("\nTemporal Grounding:")
for rank, (model, score) in enumerate(mcr_bt.get_category_rankings('temporal_grounding'), 1):
    print(f"  {rank}. {model}: {score:.3f}")

# --- Elo Example ---
print("\n=== Elo Rankings ===")
mcr_elo = MultiCategoryRanker(method='elo', k_factor=32)
mcr_elo.fit(comparisons_by_category)

mcr_elo.export_csv('elo_rankings.csv')
mcr_elo.export_json('elo_report.json')

# View category-specific rankings
print("\nTemporal Grounding (Elo):")
for rank, (model, rating) in enumerate(mcr_elo.get_category_rankings('temporal_grounding'), 1):
    print(f"  {rank}. {model}: {rating:.0f}")

# --- Single Category Example ---
print("\n=== Single Category Bradley-Terry ===")
bt = BradleyTerryRanker()
bt.fit(comparisons_by_category['temporal_grounding'])

print("Win Probabilities:")
rankings = bt.get_rankings()
models = [model for model, _ in rankings]
for i, model_i in enumerate(models):
    for model_j in models[i+1:]:
        prob = bt.get_win_probability(model_i, model_j)
        print(f"  P({model_i} beats {model_j}) = {prob:.3f}")

# Get confidence intervals
cis = bt.get_confidence_intervals(comparisons_by_category['temporal_grounding'], n_bootstrap=100)
print("\nConfidence Intervals (95%):")
for model, (ci_lower, ci_upper) in cis.items():
    score = dict(rankings)[model]
    print(f"  {model}: {score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
