"""
Ranking module for LLM video benchmark evaluation.

Implements Bradley-Terry and Elo ranking systems for pairwise tournament evaluation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BradleyTerryRanker:
    """
    Bradley-Terry model for pairwise comparisons.
    
    P(i beats j) = π_i / (π_i + π_j)
    
    Attributes:
        n_items: Number of items to rank
        strengths: Estimated strength parameters (normalized)
        item_names: List of item names
        item_to_idx: Mapping from item name to index
    """
    
    def __init__(self, item_names: Optional[List[str]] = None):
        """
        Initialize with optional item names.
        
        Args:
            item_names: Optional list of item names to pre-register
        """
        self.item_names: Optional[List[str]] = item_names
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.n_items: int = 0
        self.strengths: np.ndarray = np.array([])
        self._fitted: bool = False
    
    def fit(self, comparisons: List[Tuple[str, str]]) -> 'BradleyTerryRanker':
        """
        Fit model using maximum likelihood estimation.
        
        Args:
            comparisons: List of (winner, loser) tuples
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If comparisons list is empty
        """
        if not comparisons:
            raise ValueError("Comparisons list cannot be empty")
        
        # Extract unique items
        all_items = set()
        for winner, loser in comparisons:
            all_items.add(winner)
            all_items.add(loser)
        
        if len(all_items) < 2:
            raise ValueError("Need at least 2 distinct items to rank")
        
        # Create index mapping
        if self.item_names is None:
            self.item_names = sorted(list(all_items))
        else:
            # Add any missing items
            for item in all_items:
                if item not in self.item_names:
                    self.item_names.append(item)
        
        self.item_names = sorted(self.item_names)
        self.n_items = len(self.item_names)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_names)}
        self.idx_to_item = {idx: item for idx, item in enumerate(self.item_names)}
        
        # Convert comparisons to indices
        comparisons_idx = [
            (self.item_to_idx[winner], self.item_to_idx[loser])
            for winner, loser in comparisons
        ]
        
        # Check connectivity
        is_connected = self._is_fully_connected(comparisons_idx)
        
        # Initialize parameters (log-space for numerical stability)
        initial_params = np.zeros(self.n_items)
        
        # Optimize
        if is_connected:
            # Standard MLE
            result = minimize(
                self._neg_log_likelihood,
                initial_params,
                args=(comparisons_idx,),
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
        else:
            # MAP estimation with Gaussian prior
            logger.info("Graph not fully connected, using MAP estimation with prior")
            lambda_prior = 1.0
            result = minimize(
                self._neg_log_likelihood_with_prior,
                initial_params,
                args=(comparisons_idx, lambda_prior),
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            if not result.success:
                logger.warning(f"MAP optimization did not converge: {result.message}")
        
        # Convert from log-space to probabilities (softmax normalization)
        log_strengths = result.x
        # Normalize by subtracting max for numerical stability
        log_strengths_normalized = log_strengths - np.max(log_strengths)
        exp_strengths = np.exp(log_strengths_normalized)
        self.strengths = exp_strengths / exp_strengths.sum()
        
        self._fitted = True
        logger.info(f"Fitted Bradley-Terry model on {len(comparisons)} comparisons, "
                   f"converged in {result.nit} iterations")
        
        return self
    
    def _is_fully_connected(self, comparisons: List[Tuple[int, int]]) -> bool:
        """
        Check if comparison graph is fully connected using BFS.
        
        Args:
            comparisons: List of (winner_idx, loser_idx) tuples
            
        Returns:
            True if graph is connected, False otherwise
        """
        if not comparisons:
            return False
        
        # Build adjacency list (undirected graph)
        graph: Dict[int, List[int]] = {i: [] for i in range(self.n_items)}
        for winner, loser in comparisons:
            graph[winner].append(loser)
            graph[loser].append(winner)
        
        # BFS from first node
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all nodes were visited
        return len(visited) == self.n_items
    
    def _neg_log_likelihood(self, params: np.ndarray, 
                           comparisons: List[Tuple[int, int]]) -> float:
        """
        Compute negative log-likelihood for optimization.
        
        Args:
            params: Log-strength parameters
            comparisons: List of (winner_idx, loser_idx) tuples
            
        Returns:
            Negative log-likelihood value
        """
        log_lik = 0.0
        for winner_idx, loser_idx in comparisons:
            # log P(winner beats loser) = log(exp(θ_winner) / (exp(θ_winner) + exp(θ_loser)))
            # = θ_winner - log(exp(θ_winner) + exp(θ_loser))
            # = θ_winner - logsumexp([θ_winner, θ_loser])
            log_lik += params[winner_idx] - logsumexp([params[winner_idx], params[loser_idx]])
        
        return -log_lik  # Return negative for minimization
    
    def _neg_log_likelihood_with_prior(self, params: np.ndarray,
                                       comparisons: List[Tuple[int, int]],
                                       lambda_prior: float) -> float:
        """
        Compute negative log-likelihood with Gaussian prior (for MAP estimation).
        
        Args:
            params: Log-strength parameters
            comparisons: List of (winner_idx, loser_idx) tuples
            lambda_prior: Prior regularization strength
            
        Returns:
            Negative log-posterior value
        """
        # Log-likelihood
        log_lik = -self._neg_log_likelihood(params, comparisons)
        
        # Log-prior: -0.5 * lambda * ||params||^2
        log_prior = -0.5 * lambda_prior * np.sum(params ** 2)
        
        # Return negative log-posterior
        return -(log_lik + log_prior)
    
    def get_rankings(self, item_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Get ranked list with scores.
        
        Args:
            item_names: Optional list to filter/sort items
            
        Returns:
            List of (item_name, score) sorted by score descending
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting rankings")
        
        if item_names is None:
            item_names = self.item_names
        
        rankings = [
            (self.idx_to_item[idx], float(self.strengths[idx]))
            for idx in range(self.n_items)
            if self.idx_to_item[idx] in item_names
        ]
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_win_probability(self, item_i: str, item_j: str) -> float:
        """
        Calculate P(i beats j).
        
        Args:
            item_i: Name of first item
            item_j: Name of second item
            
        Returns:
            Probability that item_i beats item_j
            
        Raises:
            KeyError: If either item not in fitted model
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before computing win probabilities")
        
        if item_i not in self.item_to_idx or item_j not in self.item_to_idx:
            raise KeyError(f"Item not found in model: {item_i} or {item_j}")
        
        idx_i = self.item_to_idx[item_i]
        idx_j = self.item_to_idx[item_j]
        
        strength_i = self.strengths[idx_i]
        strength_j = self.strengths[idx_j]
        
        # P(i beats j) = π_i / (π_i + π_j)
        prob = strength_i / (strength_i + strength_j)
        return float(prob)
    
    def get_confidence_intervals(self, comparisons: List[Tuple[str, str]], 
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Bootstrap confidence intervals.
        
        Args:
            comparisons: Original comparison list
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dict mapping item_name -> (ci_lower, ci_upper)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before computing confidence intervals")
        
        bootstrap_strengths: Dict[str, List[float]] = {
            item: [] for item in self.item_names
        }
        
        n_comparisons = len(comparisons)
        alpha = 1 - confidence
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = [
                comparisons[np.random.randint(n_comparisons)]
                for _ in range(n_comparisons)
            ]
            
            # Fit new model
            bt_bootstrap = BradleyTerryRanker(item_names=self.item_names)
            try:
                bt_bootstrap.fit(bootstrap_sample)
                rankings = bt_bootstrap.get_rankings()
                for item, score in rankings:
                    bootstrap_strengths[item].append(score)
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {e}")
                continue
        
        # Calculate percentiles
        cis = {}
        for item in self.item_names:
            if bootstrap_strengths[item]:
                strengths_array = np.array(bootstrap_strengths[item])
                ci_lower = float(np.percentile(strengths_array, lower_percentile))
                ci_upper = float(np.percentile(strengths_array, upper_percentile))
                cis[item] = (ci_lower, ci_upper)
            else:
                # Fallback to point estimate
                rankings_dict = dict(self.get_rankings())
                point_estimate = rankings_dict.get(item, 0.0)
                cis[item] = (point_estimate, point_estimate)
        
        return cis


class EloRanker:
    """
    Elo rating system for sequential pairwise comparisons.
    
    Expected score: E = 1 / (1 + 10^((Rb - Ra) / 400))
    Rating update: New = Old + K × (Actual - Expected)
    """
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """
        Initialize Elo system.
        
        Args:
            k_factor: Maximum rating change per match (default: 32)
            initial_rating: Starting rating for new players (default: 1500)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict] = []  # Track all rating changes
    
    def add_player(self, name: str, rating: Optional[float] = None) -> None:
        """
        Add player with optional custom initial rating.
        
        Args:
            name: Player name
            rating: Optional custom initial rating (defaults to initial_rating)
        """
        if rating is None:
            rating = self.initial_rating
        self.ratings[name] = rating
    
    def record_match(self, winner: str, loser: str, 
                    actual_score: float = 1.0) -> None:
        """
        Record a match result and update ratings.
        
        Args:
            winner: Name of winning player
            loser: Name of losing player
            actual_score: Score for winner (1.0 = win, 0.5 = draw, 0.0 = loss)
        """
        # Auto-add players if they don't exist
        if winner not in self.ratings:
            self.add_player(winner)
        if loser not in self.ratings:
            self.add_player(loser)
        
        rating_winner = self.ratings[winner]
        rating_loser = self.ratings[loser]
        
        # Calculate expected scores
        expected_winner = self.expected_score(rating_winner, rating_loser)
        expected_loser = 1.0 - expected_winner
        
        # Update ratings
        # For winner: R_new = R_old + K × (actual - expected)
        # For loser: R_new = R_old + K × ((1 - actual) - expected)
        change_winner = self.k_factor * (actual_score - expected_winner)
        change_loser = self.k_factor * ((1.0 - actual_score) - expected_loser)
        
        self.ratings[winner] = rating_winner + change_winner
        self.ratings[loser] = rating_loser + change_loser
        
        # Record in history
        timestamp = datetime.now().isoformat()
        self.history.append({
            'timestamp': timestamp,
            'player': winner,
            'rating': self.ratings[winner],
            'opponent': loser,
            'result': 'win' if actual_score == 1.0 else ('draw' if actual_score == 0.5 else 'loss')
        })
        self.history.append({
            'timestamp': timestamp,
            'player': loser,
            'rating': self.ratings[loser],
            'opponent': winner,
            'result': 'loss' if actual_score == 1.0 else ('draw' if actual_score == 0.5 else 'win')
        })
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A vs player B.
        
        Formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        
        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B
            
        Returns:
            Expected score for player A (between 0 and 1)
        """
        exponent = (rating_b - rating_a) / 400.0
        expected = 1.0 / (1.0 + 10.0 ** exponent)
        return float(expected)
    
    def get_ratings(self) -> Dict[str, float]:
        """
        Get current ratings for all players.
        
        Returns:
            Dictionary mapping player name to current rating
        """
        return self.ratings.copy()
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get sorted rankings (highest to lowest).
        
        Returns:
            List of (player_name, rating) tuples sorted by rating descending
        """
        rankings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_history(self) -> pd.DataFrame:
        """
        Get rating history as DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, player, rating, opponent, result
        """
        if not self.history:
            return pd.DataFrame(columns=['timestamp', 'player', 'rating', 'opponent', 'result'])
        return pd.DataFrame(self.history)


class MultiCategoryRanker:
    """
    Aggregate pairwise comparisons across multiple categories.
    
    Supports both Bradley-Terry and Elo methods.
    """
    
    def __init__(self, method: str = 'bradley_terry', **kwargs):
        """
        Initialize multi-category ranker.
        
        Args:
            method: 'bradley_terry' or 'elo'
            **kwargs: Additional arguments for underlying ranker
                     (e.g., k_factor for Elo, n_bootstrap for BT)
        """
        if method not in ['bradley_terry', 'elo']:
            raise ValueError(f"Method must be 'bradley_terry' or 'elo', got '{method}'")
        
        self.method = method
        self.kwargs = kwargs
        self.rankers: Dict[str, BradleyTerryRanker | EloRanker] = {}
        self.comparisons_by_category: Dict[str, List[Tuple[str, str]]] = {}
        self._fitted = False
    
    def fit(self, comparisons_by_category: Dict[str, List[Tuple[str, str]]]) -> 'MultiCategoryRanker':
        """
        Fit separate models for each category.
        
        Args:
            comparisons_by_category: Dict mapping category -> comparison list
            
        Example:
            {
                'temporal_grounding': [('GPT-4V', 'Claude'), ...],
                'entity_recognition': [('GPT-4V', 'Gemini'), ...],
            }
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If any category has empty comparisons
        """
        self.comparisons_by_category = comparisons_by_category
        
        for category, comparisons in comparisons_by_category.items():
            if not comparisons:
                raise ValueError(f"Category '{category}' has empty comparisons list")
            
            if self.method == 'bradley_terry':
                ranker = BradleyTerryRanker()
                ranker.fit(comparisons)
                self.rankers[category] = ranker
            elif self.method == 'elo':
                ranker = EloRanker(
                    k_factor=self.kwargs.get('k_factor', 32),
                    initial_rating=self.kwargs.get('initial_rating', 1500)
                )
                # For Elo, we need to record matches sequentially
                for winner, loser in comparisons:
                    ranker.record_match(winner, loser)
                self.rankers[category] = ranker
        
        self._fitted = True
        return self
    
    def get_category_rankings(self, category: str) -> List[Tuple[str, float]]:
        """
        Get rankings for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of (item_name, score) sorted by score descending
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting rankings")
        
        if category not in self.rankers:
            raise ValueError(f"Category '{category}' not found")
        
        ranker = self.rankers[category]
        if isinstance(ranker, BradleyTerryRanker):
            return ranker.get_rankings()
        else:  # EloRanker
            return ranker.get_rankings()
    
    def get_overall_ranking(self, weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Compute weighted average across categories.
        
        Args:
            weights: Optional dict mapping category -> weight
                    If None, uses equal weights
                    
        Returns:
            Overall rankings sorted by aggregated score
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting overall rankings")
        
        # Get all unique items across categories
        all_items = set()
        for ranker in self.rankers.values():
            if isinstance(ranker, BradleyTerryRanker):
                rankings = ranker.get_rankings()
            else:  # EloRanker
                rankings = ranker.get_rankings()
            all_items.update([item for item, _ in rankings])
        
        # Normalize weights
        if weights is None:
            weights = {cat: 1.0 / len(self.rankers) for cat in self.rankers.keys()}
        else:
            total_weight = sum(weights.values())
            weights = {cat: w / total_weight for cat, w in weights.items()}
        
        # Aggregate scores
        aggregated_scores: Dict[str, float] = {item: 0.0 for item in all_items}
        
        for category, ranker in self.rankers.items():
            if category not in weights:
                continue
            
            weight = weights[category]
            if isinstance(ranker, BradleyTerryRanker):
                rankings = ranker.get_rankings()
                # Normalize scores to [0, 1] if needed (already normalized for BT)
                rankings_dict = dict(rankings)
            else:  # EloRanker
                rankings = ranker.get_rankings()
                # Normalize Elo ratings to [0, 1] range
                ratings_dict = dict(rankings)
                if ratings_dict:
                    min_rating = min(ratings_dict.values())
                    max_rating = max(ratings_dict.values())
                    if max_rating > min_rating:
                        rankings_dict = {
                            item: (rating - min_rating) / (max_rating - min_rating)
                            for item, rating in ratings_dict.items()
                        }
                    else:
                        rankings_dict = {item: 0.5 for item in ratings_dict.keys()}
            
            for item, score in rankings_dict.items():
                aggregated_scores[item] += weight * score
        
        # Sort by aggregated score
        overall_rankings = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        return overall_rankings
    
    def export_csv(self, filepath: str) -> None:
        """
        Export rankings to CSV.
        
        Format:
        model,overall_score,temporal_grounding,entity_recognition,...
        GPT-4V,0.85,0.87,0.83,...
        
        Args:
            filepath: Path to output CSV file
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before exporting")
        
        # Get overall rankings
        overall = self.get_overall_ranking()
        overall_dict = dict(overall)
        
        # Get category rankings
        category_data: Dict[str, Dict[str, float]] = {}
        for category in self.rankers.keys():
            rankings = self.get_category_rankings(category)
            category_data[category] = dict(rankings)
        
        # Build DataFrame
        all_models = set(overall_dict.keys())
        for cat_dict in category_data.values():
            all_models.update(cat_dict.keys())
        
        rows = []
        for model in all_models:
            row = {'model': model, 'overall_score': overall_dict.get(model, 0.0)}
            for category in self.rankers.keys():
                row[category] = category_data[category].get(model, 0.0)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('overall_score', ascending=False)
        df.to_csv(filepath, index=False)
    
    def export_json(self, filepath: str, 
                   include_comparisons: bool = True) -> None:
        """
        Export detailed report to JSON.
        
        Args:
            filepath: Path to output JSON file
            include_comparisons: Whether to include pairwise win probabilities
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before exporting")
        
        # Get overall rankings
        overall = self.get_overall_ranking()
        
        # Build overall rankings with metadata
        overall_rankings_list = []
        for rank, (model, score) in enumerate(overall, 1):
            entry = {
                'model': model,
                'score': float(score),
                'rank': rank
            }
            
            # Add confidence intervals if Bradley-Terry
            if self.method == 'bradley_terry':
                n_bootstrap = self.kwargs.get('n_bootstrap', 1000)
                # Get CI from first category that has this model
                for category, ranker in self.rankers.items():
                    if isinstance(ranker, BradleyTerryRanker):
                        try:
                            cis = ranker.get_confidence_intervals(
                                self.comparisons_by_category[category],
                                n_bootstrap=n_bootstrap
                            )
                            if model in cis:
                                ci_lower, ci_upper = cis[model]
                                entry['ci_lower'] = float(ci_lower)
                                entry['ci_upper'] = float(ci_upper)
                                break
                        except Exception:
                            pass
            
            overall_rankings_list.append(entry)
        
        # Build category rankings
        category_rankings_dict = {}
        for category in self.rankers.keys():
            rankings = self.get_category_rankings(category)
            category_rankings_dict[category] = [
                {
                    'model': model,
                    'score': float(score),
                    'rank': rank + 1
                }
                for rank, (model, score) in enumerate(rankings)
            ]
        
        # Build pairwise win probabilities (if Bradley-Terry and requested)
        pairwise_win_probabilities = {}
        if include_comparisons and self.method == 'bradley_terry':
            for category, ranker in self.rankers.items():
                if isinstance(ranker, BradleyTerryRanker):
                    rankings = ranker.get_rankings()
                    models = [model for model, _ in rankings]
                    pairwise_win_probabilities[category] = {}
                    for i, model_i in enumerate(models):
                        for model_j in models[i+1:]:
                            prob = ranker.get_win_probability(model_i, model_j)
                            key = f"{model_i}_vs_{model_j}"
                            pairwise_win_probabilities[category][key] = float(prob)
        
        # Build metadata
        all_models = set()
        for rankings in category_rankings_dict.values():
            all_models.update([entry['model'] for entry in rankings])
        
        n_comparisons_per_category = {
            category: len(comparisons)
            for category, comparisons in self.comparisons_by_category.items()
        }
        
        # Get weights (equal if not specified)
        weights = {cat: 1.0 / len(self.rankers) for cat in self.rankers.keys()}
        
        metadata = {
            'n_models': len(all_models),
            'n_comparisons_per_category': n_comparisons_per_category,
            'weights': weights
        }
        
        if self.method == 'bradley_terry':
            metadata['bootstrap_samples'] = self.kwargs.get('n_bootstrap', 1000)
        
        # Build final structure
        report = {
            'method': self.method,
            'timestamp': datetime.now().isoformat() + 'Z',
            'categories': list(self.rankers.keys()),
            'overall_rankings': overall_rankings_list,
            'category_rankings': category_rankings_dict,
            'metadata': metadata
        }
        
        if pairwise_win_probabilities:
            report['pairwise_win_probabilities'] = pairwise_win_probabilities
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
