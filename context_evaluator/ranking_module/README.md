# Ranking Module for LLM Video Benchmark Evaluation

Comprehensive guide for the ranking module used in LLM video benchmark evaluation. This module implements Bradley-Terry and Elo ranking systems for pairwise tournament evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)
6. [Integration Guide](#integration-guide)
7. [Example Usage](#example-usage)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Testing](#testing)

---

## Overview

The ranking module provides statistical ranking systems for evaluating LLM performance based on pairwise comparisons from a judge LLM. It supports:

- **Bradley-Terry Model**: Maximum likelihood estimation with bootstrap confidence intervals
- **Elo Rating System**: Sequential pairwise comparisons with configurable parameters
- **Multi-Category Ranking**: Aggregate rankings across multiple evaluation dimensions
- **Export Formats**: CSV (quick viewing) and JSON (detailed analysis)

### Key Features

- Handles sparse/disconnected comparison graphs
- Bootstrap confidence intervals for uncertainty quantification
- Weighted aggregation across categories
- Full rating history tracking (Elo)
- Comprehensive export functionality

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy>=1.24.0` - Numerical computations
- `pandas>=2.0.0` - Data manipulation and export
- `scipy>=1.11.0` - Optimization algorithms
- `pytest>=7.4.0` - Testing framework (dev dependency)

### Verify Installation

```python
from ranking import BradleyTerryRanker, EloRanker, MultiCategoryRanker
print("Installation successful!")
```

---

## Quick Start

### Basic Single-Category Ranking

```python
from ranking import BradleyTerryRanker

# Pairwise comparisons: (winner, loser)
comparisons = [
    ('GPT-4V', 'Claude-3.5'),
    ('GPT-4V', 'Gemini'),
    ('Claude-3.5', 'Gemini'),
]

# Fit model
bt = BradleyTerryRanker()
bt.fit(comparisons)

# Get rankings
rankings = bt.get_rankings()
for model, score in rankings:
    print(f"{model}: {score:.3f}")

# Get win probabilities
prob = bt.get_win_probability('GPT-4V', 'Claude-3.5')
print(f"P(GPT-4V beats Claude-3.5) = {prob:.3f}")

# Get confidence intervals
cis = bt.get_confidence_intervals(comparisons, n_bootstrap=1000)
```

### Multi-Category Ranking

```python
from ranking import MultiCategoryRanker

# Comparisons organized by category
comparisons_by_category = {
    'temporal_grounding': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
    ],
    'entity_recognition': [
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
    ],
}

# Fit with Bradley-Terry
mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)
mcr.fit(comparisons_by_category)

# Export results
mcr.export_csv('rankings.csv')
mcr.export_json('report.json')

# Get overall rankings
overall = mcr.get_overall_ranking()

# Get category-specific rankings
cat_rankings = mcr.get_category_rankings('temporal_grounding')

# Custom weights
weighted = mcr.get_overall_ranking(weights={
    'temporal_grounding': 0.6,
    'entity_recognition': 0.4,
})
```

### Elo Rating System

```python
from ranking import EloRanker

elo = EloRanker(k_factor=32, initial_rating=1500)

# Record matches sequentially
elo.record_match('GPT-4V', 'Claude-3.5')
elo.record_match('GPT-4V', 'Gemini')
elo.record_match('Claude-3.5', 'Gemini')

# Get current ratings
ratings = elo.get_ratings()

# Get rankings
rankings = elo.get_rankings()

# Get history
history_df = elo.get_history()
```

---

## API Reference

### BradleyTerryRanker

Implements the Bradley-Terry model for pairwise comparisons.

#### Methods

##### `__init__(item_names: Optional[List[str]] = None)`

Initialize the ranker.

**Parameters:**
- `item_names` (optional): Pre-register item names

**Example:**
```python
bt = BradleyTerryRanker()
# or
bt = BradleyTerryRanker(item_names=['GPT-4V', 'Claude-3.5', 'Gemini'])
```

##### `fit(comparisons: List[Tuple[str, str]]) -> BradleyTerryRanker`

Fit model using maximum likelihood estimation.

**Parameters:**
- `comparisons`: List of `(winner, loser)` tuples

**Returns:** `self` (for method chaining)

**Raises:**
- `ValueError`: If comparisons list is empty or has < 2 distinct items

**Example:**
```python
comparisons = [('GPT-4V', 'Claude-3.5'), ('GPT-4V', 'Gemini')]
bt.fit(comparisons)
```

##### `get_rankings(item_names: Optional[List[str]] = None) -> List[Tuple[str, float]]`

Get ranked list with scores (normalized probabilities).

**Parameters:**
- `item_names` (optional): Filter/sort specific items

**Returns:** List of `(item_name, score)` tuples, sorted by score descending

**Example:**
```python
rankings = bt.get_rankings()
# [('GPT-4V', 0.45), ('Claude-3.5', 0.32), ('Gemini', 0.23)]
```

##### `get_win_probability(item_i: str, item_j: str) -> float`

Calculate P(i beats j) using Bradley-Terry formula.

**Parameters:**
- `item_i`: Name of first item
- `item_j`: Name of second item

**Returns:** Probability between 0 and 1

**Example:**
```python
prob = bt.get_win_probability('GPT-4V', 'Claude-3.5')
# Returns: 0.72 (72% chance GPT-4V beats Claude-3.5)
```

##### `get_confidence_intervals(comparisons: List[Tuple[str, str]], n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]`

Bootstrap confidence intervals for strength estimates.

**Parameters:**
- `comparisons`: Original comparison list
- `n_bootstrap`: Number of bootstrap samples (default: 1000)
- `confidence`: Confidence level (default: 0.95 for 95% CI)

**Returns:** Dict mapping `item_name -> (ci_lower, ci_upper)`

**Example:**
```python
cis = bt.get_confidence_intervals(comparisons, n_bootstrap=1000)
# {'GPT-4V': (0.38, 0.52), 'Claude-3.5': (0.25, 0.38), ...}
```

---

### EloRanker

Implements the Elo rating system for sequential pairwise comparisons.

#### Methods

##### `__init__(k_factor: float = 32, initial_rating: float = 1500)`

Initialize Elo system.

**Parameters:**
- `k_factor`: Maximum rating change per match (default: 32)
- `initial_rating`: Starting rating for new players (default: 1500)

**Example:**
```python
elo = EloRanker(k_factor=32, initial_rating=1500)
```

##### `add_player(name: str, rating: Optional[float] = None) -> None`

Add player with optional custom initial rating.

**Parameters:**
- `name`: Player name
- `rating` (optional): Custom initial rating

**Example:**
```python
elo.add_player('GPT-4V')
elo.add_player('Claude-3.5', rating=1600)  # Custom rating
```

##### `record_match(winner: str, loser: str, actual_score: float = 1.0) -> None`

Record a match result and update ratings.

**Parameters:**
- `winner`: Name of winning player
- `loser`: Name of losing player
- `actual_score`: Score for winner (1.0 = win, 0.5 = draw, 0.0 = loss)

**Note:** Players are automatically added if they don't exist.

**Example:**
```python
elo.record_match('GPT-4V', 'Claude-3.5')  # GPT-4V wins
elo.record_match('GPT-4V', 'Gemini', actual_score=0.5)  # Draw
```

##### `expected_score(rating_a: float, rating_b: float) -> float`

Calculate expected score for player A vs player B.

**Formula:** `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

**Parameters:**
- `rating_a`: Rating of player A
- `rating_b`: Rating of player B

**Returns:** Expected score for player A (between 0 and 1)

**Example:**
```python
expected = elo.expected_score(1600, 1500)
# Returns: ~0.64 (64% expected score for 1600-rated player)
```

##### `get_ratings() -> Dict[str, float]`

Get current ratings for all players.

**Returns:** Dictionary mapping player name to current rating

**Example:**
```python
ratings = elo.get_ratings()
# {'GPT-4V': 1523.2, 'Claude-3.5': 1476.8, ...}
```

##### `get_rankings() -> List[Tuple[str, float]]`

Get sorted rankings (highest to lowest).

**Returns:** List of `(player_name, rating)` tuples

**Example:**
```python
rankings = elo.get_rankings()
# [('GPT-4V', 1523.2), ('Claude-3.5', 1476.8), ...]
```

##### `get_history() -> pd.DataFrame`

Get rating history as DataFrame.

**Returns:** DataFrame with columns: `timestamp`, `player`, `rating`, `opponent`, `result`

**Example:**
```python
history = elo.get_history()
print(history.head())
```

---

### MultiCategoryRanker

Aggregates pairwise comparisons across multiple evaluation categories.

#### Methods

##### `__init__(method: str = 'bradley_terry', **kwargs)`

Initialize multi-category ranker.

**Parameters:**
- `method`: `'bradley_terry'` or `'elo'`
- `**kwargs`: Additional arguments:
  - For Bradley-Terry: `n_bootstrap` (default: 1000)
  - For Elo: `k_factor` (default: 32), `initial_rating` (default: 1500)

**Example:**
```python
# Bradley-Terry
mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)

# Elo
mcr = MultiCategoryRanker(method='elo', k_factor=32, initial_rating=1500)
```

##### `fit(comparisons_by_category: Dict[str, List[Tuple[str, str]]]) -> MultiCategoryRanker`

Fit separate models for each category.

**Parameters:**
- `comparisons_by_category`: Dict mapping `category -> comparison list`

**Returns:** `self` (for method chaining)

**Raises:**
- `ValueError`: If any category has empty comparisons

**Example:**
```python
comparisons = {
    'temporal_grounding': [('GPT-4V', 'Claude-3.5'), ...],
    'entity_recognition': [('GPT-4V', 'Gemini'), ...],
}
mcr.fit(comparisons)
```

##### `get_category_rankings(category: str) -> List[Tuple[str, float]]`

Get rankings for a specific category.

**Parameters:**
- `category`: Category name

**Returns:** List of `(item_name, score)` sorted by score descending

**Example:**
```python
rankings = mcr.get_category_rankings('temporal_grounding')
```

##### `get_overall_ranking(weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]`

Compute weighted average across categories.

**Parameters:**
- `weights` (optional): Dict mapping `category -> weight`. If None, uses equal weights.

**Returns:** Overall rankings sorted by aggregated score

**Example:**
```python
# Equal weights
overall = mcr.get_overall_ranking()

# Custom weights
overall = mcr.get_overall_ranking(weights={
    'temporal_grounding': 0.4,
    'entity_recognition': 0.3,
    'clarification_questions': 0.2,
    'answer_quality': 0.1,
})
```

##### `export_csv(filepath: str) -> None`

Export rankings to CSV format.

**Parameters:**
- `filepath`: Path to output CSV file

**CSV Format:**
```csv
model,overall_score,temporal_grounding,entity_recognition,...
GPT-4V,0.85,0.87,0.83,...
Claude-3.5,0.72,0.74,0.71,...
```

**Example:**
```python
mcr.export_csv('rankings.csv')
```

##### `export_json(filepath: str, include_comparisons: bool = True) -> None`

Export detailed report to JSON format.

**Parameters:**
- `filepath`: Path to output JSON file
- `include_comparisons`: Whether to include pairwise win probabilities

**JSON Structure:**
```json
{
  "method": "bradley_terry",
  "timestamp": "2025-01-18T14:30:00Z",
  "categories": ["temporal_grounding", ...],
  "overall_rankings": [
    {
      "model": "GPT-4V",
      "score": 0.85,
      "rank": 1,
      "ci_lower": 0.78,
      "ci_upper": 0.92
    },
    ...
  ],
  "category_rankings": {...},
  "pairwise_win_probabilities": {...},
  "metadata": {...}
}
```

**Example:**
```python
mcr.export_json('report.json', include_comparisons=True)
```

---

## Implementation Details

### Bradley-Terry Model

#### Mathematical Foundation

The Bradley-Terry model assumes that the probability that item `i` beats item `j` is:

```
P(i beats j) = π_i / (π_i + π_j)
```

where `π_i` and `π_j` are strength parameters for items `i` and `j`.

#### Optimization

1. **Log-space parameterization**: Uses `θ_i = log(π_i)` for numerical stability
2. **Maximum Likelihood Estimation**: Minimizes negative log-likelihood:
   ```
   -log L = -Σ[θ_winner - log(exp(θ_winner) + exp(θ_loser))]
   ```
3. **Normalization**: Converts to probabilities using softmax: `π_i = exp(θ_i) / Σ exp(θ_j)`

#### Handling Sparse Graphs

If the comparison graph is not fully connected (disconnected components):
- Uses **MAP estimation** with Gaussian prior: `log p(θ) = -0.5 * λ * ||θ||²`
- Regularization strength `λ = 1.0` (configurable)
- Ensures stable estimates even with limited comparisons

#### Bootstrap Confidence Intervals

1. Resample comparisons with replacement (default: 1000 iterations)
2. Fit new model on each bootstrap sample
3. Calculate percentiles: `(1-confidence)/2` and `1-(1-confidence)/2`
4. Returns confidence interval for each item's strength

### Elo Rating System

#### Rating Update Formula

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_new = R_old + K × (Actual - Expected)
```

Where:
- `E_A`: Expected score for player A
- `R_A`, `R_B`: Current ratings
- `K`: K-factor (maximum rating change per match)
- `Actual`: Actual match result (1.0 = win, 0.5 = draw, 0.0 = loss)

#### K-Factor Selection

- **Default: 32** - Standard for most competitions
- **Higher K (e.g., 64)**: Faster rating changes, more responsive to recent performance
- **Lower K (e.g., 16)**: Slower changes, more stable ratings

#### Rating Normalization

For multi-category aggregation, Elo ratings are normalized to [0, 1] range:
```python
normalized = (rating - min_rating) / (max_rating - min_rating)
```

This ensures fair aggregation with Bradley-Terry scores (which are already probabilities).

### Multi-Category Aggregation

#### Weighted Average

Overall score for model `m`:
```
score_m = Σ (weight_c × score_m,c)
```

Where:
- `weight_c`: Weight for category `c` (normalized to sum to 1)
- `score_m,c`: Normalized score for model `m` in category `c`

#### Category Weights

- **Equal weights** (default): `1 / n_categories` for each category
- **Custom weights**: User-specified, automatically normalized
- **Best practice**: Use domain knowledge to set weights based on category importance

---

## Integration Guide

### Integration with Judge LLM Pipeline

The ranking module expects pairwise comparisons from a judge LLM. Here's how to integrate:

#### Step 1: Collect Pairwise Comparisons

```python
# Example: Judge LLM returns comparisons
def get_judge_comparisons(video_id, models, category):
    """
    Get pairwise comparisons from judge LLM.
    
    Returns:
        List of (winner, loser) tuples
    """
    comparisons = []
    
    # Compare all pairs
    for i, model_i in enumerate(models):
        for model_j in models[i+1:]:
            # Get responses
            response_i = get_model_response(model_i, video_id)
            response_j = get_model_response(model_j, video_id)
            
            # Judge comparison
            winner = judge_llm.compare(response_i, response_j, category)
            
            if winner == model_i:
                comparisons.append((model_i, model_j))
            elif winner == model_j:
                comparisons.append((model_j, model_i))
            # Skip if tie (or handle separately)
    
    return comparisons
```

#### Step 2: Organize by Category

```python
# Collect comparisons for each category
comparisons_by_category = {
    'temporal_grounding': [],
    'entity_recognition': [],
    'clarification_questions': [],
    'answer_quality': [],
}

for video_id in video_dataset:
    for category in comparisons_by_category.keys():
        comparisons = get_judge_comparisons(video_id, models, category)
        comparisons_by_category[category].extend(comparisons)
```

#### Step 3: Rank and Export

```python
from ranking import MultiCategoryRanker

# Fit model
mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)
mcr.fit(comparisons_by_category)

# Export results
mcr.export_csv('benchmark_rankings.csv')
mcr.export_json('benchmark_report.json', include_comparisons=True)

# Get rankings programmatically
overall = mcr.get_overall_ranking()
print("Top model:", overall[0][0])
```

### Integration with Evaluation Pipeline

```python
# Complete evaluation pipeline
class LLMBenchmarkEvaluator:
    def __init__(self, models, categories):
        self.models = models
        self.categories = categories
        self.comparisons_by_category = {cat: [] for cat in categories}
    
    def evaluate(self, video_dataset):
        """Run full evaluation pipeline."""
        # 1. Collect comparisons
        for video in video_dataset:
            for category in self.categories:
                comparisons = self._get_comparisons(video, category)
                self.comparisons_by_category[category].extend(comparisons)
        
        # 2. Rank models
        mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)
        mcr.fit(self.comparisons_by_category)
        
        # 3. Store results
        self.rankings = mcr
        return mcr
    
    def export_results(self, output_dir):
        """Export all results."""
        self.rankings.export_csv(f'{output_dir}/rankings.csv')
        self.rankings.export_json(f'{output_dir}/report.json')
    
    def _get_comparisons(self, video, category):
        # Implementation details...
        pass
```

### Handling Ties and Uncertain Comparisons

```python
# Option 1: Skip ties
comparisons = []
if judge_result == 'model_a':
    comparisons.append(('model_a', 'model_b'))
elif judge_result == 'model_b':
    comparisons.append(('model_b', 'model_a'))
# Skip if tie

# Option 2: Use Elo with draws
elo = EloRanker()
if judge_result == 'tie':
    elo.record_match('model_a', 'model_b', actual_score=0.5)
else:
    winner, loser = judge_result.split('_beats_')
    elo.record_match(winner, loser)
```

### Batch Processing

```python
# Process comparisons in batches
def process_batch(comparisons_batch, method='bradley_terry'):
    """Process a batch of comparisons."""
    if method == 'bradley_terry':
        bt = BradleyTerryRanker()
        bt.fit(comparisons_batch)
        return bt.get_rankings()
    else:
        elo = EloRanker()
        for winner, loser in comparisons_batch:
            elo.record_match(winner, loser)
        return elo.get_rankings()

# Process multiple batches
all_rankings = []
for batch in comparison_batches:
    rankings = process_batch(batch)
    all_rankings.append(rankings)
```

---

## Example Usage

### Example 1: Complete Tournament Evaluation

```python
from ranking import MultiCategoryRanker

# Realistic tournament data
comparisons_by_category = {
    'temporal_grounding': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
        ('GPT-4V', 'Claude-3.5'),  # Multiple comparisons
        ('GPT-4V', 'Gemini'),
    ],
    'entity_recognition': [
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
        ('GPT-4V', 'Claude-3.5'),
        ('Claude-3.5', 'Gemini'),
    ],
    'clarification_questions': [
        ('GPT-4V', 'Claude-3.5'),
        ('Gemini', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
    ],
    'answer_quality': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
        ('GPT-4V', 'Claude-3.5'),
    ]
}

# Fit with Bradley-Terry
mcr_bt = MultiCategoryRanker(method='bradley_terry', n_bootstrap=1000)
mcr_bt.fit(comparisons_by_category)

# Export results
mcr_bt.export_csv('bt_rankings.csv')
mcr_bt.export_json('bt_report.json')

# View rankings
print("=== Overall Rankings (Bradley-Terry) ===")
overall = mcr_bt.get_overall_ranking()
for rank, (model, score) in enumerate(overall, 1):
    print(f"{rank}. {model}: {score:.3f}")

# View category-specific rankings
print("\n=== Temporal Grounding ===")
temporal = mcr_bt.get_category_rankings('temporal_grounding')
for rank, (model, score) in enumerate(temporal, 1):
    print(f"{rank}. {model}: {score:.3f}")

# Compare with Elo
mcr_elo = MultiCategoryRanker(method='elo', k_factor=32)
mcr_elo.fit(comparisons_by_category)
mcr_elo.export_csv('elo_rankings.csv')

print("\n=== Overall Rankings (Elo) ===")
overall_elo = mcr_elo.get_overall_ranking()
for rank, (model, rating) in enumerate(overall_elo, 1):
    print(f"{rank}. {model}: {rating:.0f}")
```

### Example 2: Single Category with Confidence Intervals

```python
from ranking import BradleyTerryRanker

# Single category comparisons
comparisons = [
    ('GPT-4V', 'Claude-3.5'),
    ('GPT-4V', 'Gemini'),
    ('Claude-3.5', 'Gemini'),
    ('GPT-4V', 'Claude-3.5'),
    ('GPT-4V', 'Gemini'),
]

# Fit model
bt = BradleyTerryRanker()
bt.fit(comparisons)

# Get rankings
rankings = bt.get_rankings()
print("Rankings:")
for model, score in rankings:
    print(f"  {model}: {score:.3f}")

# Get win probabilities
print("\nWin Probabilities:")
models = [model for model, _ in rankings]
for i, model_i in enumerate(models):
    for model_j in models[i+1:]:
        prob = bt.get_win_probability(model_i, model_j)
        print(f"  P({model_i} beats {model_j}) = {prob:.3f}")

# Get confidence intervals
cis = bt.get_confidence_intervals(comparisons, n_bootstrap=1000)
print("\n95% Confidence Intervals:")
for model, (ci_lower, ci_upper) in cis.items():
    score = dict(rankings)[model]
    print(f"  {model}: {score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Example 3: Elo with History Tracking

```python
from ranking import EloRanker

elo = EloRanker(k_factor=32, initial_rating=1500)

# Record matches sequentially
matches = [
    ('GPT-4V', 'Claude-3.5'),
    ('GPT-4V', 'Gemini'),
    ('Claude-3.5', 'Gemini'),
    ('GPT-4V', 'Claude-3.5'),
    ('GPT-4V', 'Gemini'),
]

for winner, loser in matches:
    elo.record_match(winner, loser)

# Get current ratings
ratings = elo.get_ratings()
print("Current Ratings:")
for model, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {rating:.1f}")

# Get rankings
rankings = elo.get_rankings()
print("\nRankings:")
for rank, (model, rating) in enumerate(rankings, 1):
    print(f"  {rank}. {model}: {rating:.1f}")

# Get history
history = elo.get_history()
print(f"\nMatch History ({len(history)} entries):")
print(history.head(10))
```

### Example 4: Weighted Multi-Category Ranking

```python
from ranking import MultiCategoryRanker

comparisons_by_category = {
    'temporal_grounding': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
    ],
    'entity_recognition': [
        ('GPT-4V', 'Gemini'),
        ('Claude-3.5', 'Gemini'),
    ],
    'answer_quality': [
        ('GPT-4V', 'Claude-3.5'),
        ('GPT-4V', 'Gemini'),
    ],
}

mcr = MultiCategoryRanker(method='bradley_terry')
mcr.fit(comparisons_by_category)

# Equal weights (default)
overall_equal = mcr.get_overall_ranking()
print("Equal Weights:")
for model, score in overall_equal:
    print(f"  {model}: {score:.3f}")

# Custom weights (emphasize temporal_grounding)
weights = {
    'temporal_grounding': 0.5,
    'entity_recognition': 0.3,
    'answer_quality': 0.2,
}
overall_weighted = mcr.get_overall_ranking(weights=weights)
print("\nCustom Weights (temporal_grounding=0.5):")
for model, score in overall_weighted:
    print(f"  {model}: {score:.3f}")
```

### Example 5: Reading and Analyzing Exported Results

```python
import pandas as pd
import json

# Read CSV
df = pd.read_csv('rankings.csv')
print("CSV Rankings:")
print(df)

# Read JSON
with open('report.json') as f:
    report = json.load(f)

print("\nJSON Report Structure:")
print(f"Method: {report['method']}")
print(f"Timestamp: {report['timestamp']}")
print(f"Categories: {report['categories']}")

print("\nOverall Rankings:")
for entry in report['overall_rankings']:
    print(f"  {entry['rank']}. {entry['model']}: {entry['score']:.3f}")
    if 'ci_lower' in entry:
        print(f"      CI: [{entry['ci_lower']:.3f}, {entry['ci_upper']:.3f}]")

print("\nCategory Rankings:")
for category, rankings in report['category_rankings'].items():
    print(f"  {category}:")
    for entry in rankings[:3]:  # Top 3
        print(f"    {entry['rank']}. {entry['model']}: {entry['score']:.3f}")

print("\nMetadata:")
meta = report['metadata']
print(f"  Models: {meta['n_models']}")
print(f"  Comparisons per category: {meta['n_comparisons_per_category']}")
```

---

## Best Practices

### 1. Comparison Collection

- **Complete tournaments**: Ensure all pairs are compared (or at least a connected graph)
- **Multiple comparisons**: Include multiple comparisons per pair for robustness
- **Consistent categories**: Use consistent category names across evaluations

### 2. Method Selection

- **Bradley-Terry**: 
  - Use when you have all comparisons at once
  - Better for statistical inference (confidence intervals)
  - Handles sparse graphs with MAP estimation
  
- **Elo**:
  - Use when comparisons arrive sequentially
  - Better for dynamic/ongoing evaluations
  - Provides rating history

### 3. Bootstrap Confidence Intervals

- **Sample size**: Use `n_bootstrap >= 1000` for stable intervals
- **Computational cost**: Bootstrap can be slow for large datasets
- **Interpretation**: Wider intervals indicate more uncertainty

### 4. Category Weights

- **Domain knowledge**: Set weights based on category importance
- **Normalization**: Weights are automatically normalized (sum to 1)
- **Sensitivity analysis**: Test different weight configurations

### 5. Export Formats

- **CSV**: Use for quick viewing and spreadsheet analysis
- **JSON**: Use for detailed analysis, programmatic access, and reporting
- **Include comparisons**: Set `include_comparisons=True` for full reproducibility

### 6. Error Handling

```python
try:
    bt.fit(comparisons)
except ValueError as e:
    print(f"Error: {e}")
    # Handle empty comparisons, insufficient data, etc.
```

### 7. Performance Considerations

- **Large datasets**: Bootstrap can be slow; consider reducing `n_bootstrap` for quick iterations
- **Memory**: Bradley-Terry stores all comparisons; Elo only stores current ratings
- **Parallelization**: Bootstrap iterations can be parallelized (future enhancement)

---

## Troubleshooting

### Common Issues

#### 1. Empty Comparisons Error

**Error:**
```
ValueError: Comparisons list cannot be empty
```

**Solution:**
- Ensure comparisons list is not empty
- Check data collection pipeline

#### 2. Disconnected Graph Warning

**Warning:**
```
Graph not fully connected, using MAP estimation with prior
```

**Solution:**
- This is handled automatically with MAP estimation
- Add more comparisons to connect all items if possible

#### 3. Bootstrap Takes Too Long

**Issue:** Bootstrap confidence intervals are slow

**Solution:**
- Reduce `n_bootstrap` (e.g., 100 instead of 1000) for quick iterations
- Use full bootstrap only for final results

#### 4. Inconsistent Rankings Between Methods

**Issue:** Bradley-Terry and Elo produce different rankings

**Solution:**
- This is expected; Elo is sequential, Bradley-Terry is batch
- Use same comparison order for Elo if comparing directly
- Both methods should agree on the top performer with sufficient data

#### 5. Export File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
- Ensure output directory exists
- Use absolute paths or create directories first

```python
from pathlib import Path
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)
mcr.export_csv(str(output_dir / 'rankings.csv'))
```

### Debugging Tips

1. **Check comparison format**: Ensure tuples are `(winner, loser)`
2. **Verify model names**: Consistent naming across comparisons
3. **Inspect rankings**: Print intermediate results to verify logic
4. **Test with simple data**: Start with 2-3 items before scaling up

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_ranking.py -v

# Run specific test class
pytest test_ranking.py::TestBradleyTerry -v

# Run with coverage
pytest test_ranking.py --cov=ranking --cov-report=html
```

The test suite includes:
- Unit tests for each ranking method
- Edge case handling (empty comparisons, disconnected graphs)
- Realistic tournament scenarios with 3 LLMs
- Export functionality validation
- Method consistency checks

---

## Additional Resources

- **Test Suite**: See `test_ranking.py` for comprehensive examples
- **Example Script**: See `example_usage.py` for complete usage examples
- **Mathematical Details**: See implementation comments in `ranking.py`

---

## License

Part of the ECCV context-gathering benchmark research project.
