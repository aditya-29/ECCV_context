"""
Comprehensive test suite for ranking module.

Tests Bradley-Terry, Elo, and Multi-Category ranking implementations.
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from ranking import BradleyTerryRanker, EloRanker, MultiCategoryRanker


class TestBradleyTerry:
    """Test Bradley-Terry implementation."""
    
    def test_simple_ranking(self):
        """Test on simple dataset with known outcome."""
        # A beats B, A beats C, B beats C → A > B > C
        comparisons = [
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'C'),
        ]
        bt = BradleyTerryRanker()
        bt.fit(comparisons)
        rankings = bt.get_rankings()
        
        # Verify order
        assert rankings[0][0] == 'A'
        assert rankings[1][0] == 'B'
        assert rankings[2][0] == 'C'
        
        # Verify scores are normalized (sum to 1)
        scores = [score for _, score in rankings]
        assert np.isclose(sum(scores), 1.0)
    
    def test_win_probability(self):
        """Test pairwise win probability calculation."""
        comparisons = [('A', 'B'), ('A', 'B'), ('A', 'B')]
        bt = BradleyTerryRanker()
        bt.fit(comparisons)
        
        prob_a_beats_b = bt.get_win_probability('A', 'B')
        assert 0.5 < prob_a_beats_b < 1.0  # A should be favored
        
        # Probabilities should sum to 1
        prob_b_beats_a = bt.get_win_probability('B', 'A')
        assert np.isclose(prob_a_beats_b + prob_b_beats_a, 1.0)
    
    def test_confidence_intervals(self):
        """Test bootstrap CI generation."""
        comparisons = [
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'C'),
        ]
        bt = BradleyTerryRanker()
        bt.fit(comparisons)
        cis = bt.get_confidence_intervals(comparisons, n_bootstrap=100)
        
        # Check CI format
        assert 'A' in cis
        ci_lower, ci_upper = cis['A']
        assert ci_lower < ci_upper
        
        # CI should contain point estimate
        rankings = dict(bt.get_rankings())
        assert ci_lower <= rankings['A'] <= ci_upper
    
    def test_empty_comparisons(self):
        """Test error handling for empty input."""
        bt = BradleyTerryRanker()
        with pytest.raises(ValueError):
            bt.fit([])
    
    def test_single_comparison(self):
        """Test edge case of single comparison."""
        comparisons = [('A', 'B')]
        bt = BradleyTerryRanker()
        bt.fit(comparisons)
        rankings = bt.get_rankings()
        
        assert len(rankings) == 2
        assert rankings[0][0] == 'A'  # Winner ranked first
    
    def test_item_names_initialization(self):
        """Test initialization with item names."""
        item_names = ['A', 'B', 'C']
        bt = BradleyTerryRanker(item_names=item_names)
        comparisons = [('A', 'B'), ('B', 'C')]
        bt.fit(comparisons)
        rankings = bt.get_rankings()
        
        # All items should be in rankings
        ranked_items = [item for item, _ in rankings]
        assert set(ranked_items) == set(item_names)
    
    def test_disconnected_graph(self):
        """Test handling of disconnected comparison graph."""
        # Two separate groups: {A, B} and {C, D}
        comparisons = [
            ('A', 'B'),
            ('C', 'D'),
        ]
        bt = BradleyTerryRanker()
        # Should handle gracefully with MAP estimation
        bt.fit(comparisons)
        rankings = bt.get_rankings()
        
        # Should still produce rankings for all items
        ranked_items = [item for item, _ in rankings]
        assert len(ranked_items) == 4
        assert set(ranked_items) == {'A', 'B', 'C', 'D'}


class TestElo:
    """Test Elo implementation."""
    
    def test_rating_update(self):
        """Test basic rating update mechanics."""
        elo = EloRanker(k_factor=32, initial_rating=1500)
        elo.add_player('A')
        elo.add_player('B')
        
        initial_a = elo.get_ratings()['A']
        initial_b = elo.get_ratings()['B']
        
        # Equal ratings → equal expected scores → winner gains K/2
        elo.record_match('A', 'B')
        
        updated_a = elo.get_ratings()['A']
        updated_b = elo.get_ratings()['B']
        
        # A should gain exactly 16 points, B should lose 16
        assert np.isclose(updated_a - initial_a, 16)
        assert np.isclose(initial_b - updated_b, 16)
    
    def test_expected_score(self):
        """Test expected score calculation."""
        elo = EloRanker()
        
        # Equal ratings → 50% expected
        assert np.isclose(elo.expected_score(1500, 1500), 0.5)
        
        # 400 point difference → ~91% expected
        assert 0.90 < elo.expected_score(1900, 1500) < 0.92
    
    def test_rating_history(self):
        """Test history tracking."""
        elo = EloRanker()
        elo.record_match('A', 'B')
        elo.record_match('B', 'C')
        
        history = elo.get_history()
        assert len(history) >= 2
        assert 'player' in history.columns
        assert 'rating' in history.columns
    
    def test_auto_add_players(self):
        """Test automatic player addition on match recording."""
        elo = EloRanker()
        elo.record_match('A', 'B')
        
        ratings = elo.get_ratings()
        assert 'A' in ratings
        assert 'B' in ratings
        assert ratings['A'] > ratings['B']
    
    def test_custom_initial_rating(self):
        """Test custom initial rating."""
        elo = EloRanker(initial_rating=2000)
        elo.add_player('A', rating=1800)
        elo.add_player('B')  # Should use default 2000
        
        ratings = elo.get_ratings()
        assert ratings['A'] == 1800
        assert ratings['B'] == 2000
    
    def test_draw_result(self):
        """Test handling of draw (actual_score=0.5)."""
        elo = EloRanker(k_factor=32)
        elo.add_player('A')
        elo.add_player('B')
        
        initial_a = elo.get_ratings()['A']
        initial_b = elo.get_ratings()['B']
        
        # Draw: both get 0.5 points
        elo.record_match('A', 'B', actual_score=0.5)
        
        updated_a = elo.get_ratings()['A']
        updated_b = elo.get_ratings()['B']
        
        # Both should have small changes (expected score was 0.5, actual was 0.5)
        # So change should be ~0
        assert abs(updated_a - initial_a) < 1.0
        assert abs(updated_b - initial_b) < 1.0
    
    def test_get_rankings(self):
        """Test ranking retrieval."""
        elo = EloRanker()
        elo.record_match('A', 'B')
        elo.record_match('A', 'C')
        
        rankings = elo.get_rankings()
        assert len(rankings) == 3
        assert rankings[0][0] == 'A'  # Highest rating
        assert rankings[0][1] > rankings[1][1]  # A's rating > B's rating


class TestMultiCategory:
    """Test multi-category ranker."""
    
    def test_category_aggregation(self):
        """Test aggregation across categories."""
        comparisons = {
            'cat1': [('A', 'B'), ('A', 'C')],
            'cat2': [('B', 'A'), ('B', 'C')],
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        mcr.fit(comparisons)
        
        # Check per-category rankings
        cat1_rankings = mcr.get_category_rankings('cat1')
        assert cat1_rankings[0][0] == 'A'  # A wins cat1
        
        cat2_rankings = mcr.get_category_rankings('cat2')
        assert cat2_rankings[0][0] == 'B'  # B wins cat2
        
        # Overall should balance
        overall = mcr.get_overall_ranking()
        assert len(overall) == 3
    
    def test_weighted_aggregation(self):
        """Test custom weights for categories."""
        comparisons = {
            'cat1': [('A', 'B')],
            'cat2': [('B', 'A')],
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        mcr.fit(comparisons)
        
        # Equal weights
        overall_equal = mcr.get_overall_ranking()
        
        # Heavy weight on cat1 → A should win
        overall_weighted = mcr.get_overall_ranking(weights={'cat1': 0.9, 'cat2': 0.1})
        assert overall_weighted[0][0] == 'A'
    
    def test_csv_export(self, tmp_path):
        """Test CSV export format."""
        comparisons = {
            'cat1': [('A', 'B')],
            'cat2': [('B', 'C')],
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        mcr.fit(comparisons)
        
        csv_file = tmp_path / "rankings.csv"
        mcr.export_csv(str(csv_file))
        
        # Verify file exists and has correct columns
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert 'model' in df.columns
        assert 'overall_score' in df.columns
        assert 'cat1' in df.columns
        assert 'cat2' in df.columns
    
    def test_json_export(self, tmp_path):
        """Test JSON export format."""
        comparisons = {
            'cat1': [('A', 'B')],
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        mcr.fit(comparisons)
        
        json_file = tmp_path / "report.json"
        mcr.export_json(str(json_file))
        
        # Verify structure
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        
        assert 'method' in data
        assert 'overall_rankings' in data
        assert 'category_rankings' in data
        assert 'metadata' in data
    
    def test_elo_method(self):
        """Test multi-category with Elo method."""
        comparisons = {
            'cat1': [('A', 'B'), ('A', 'C')],
            'cat2': [('B', 'A'), ('B', 'C')],
        }
        
        mcr = MultiCategoryRanker(method='elo', k_factor=32)
        mcr.fit(comparisons)
        
        cat1_rankings = mcr.get_category_rankings('cat1')
        assert len(cat1_rankings) == 3
        
        overall = mcr.get_overall_ranking()
        assert len(overall) == 3
    
    def test_json_export_with_comparisons(self, tmp_path):
        """Test JSON export with pairwise comparisons included."""
        comparisons = {
            'cat1': [('A', 'B')],
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        mcr.fit(comparisons)
        
        json_file = tmp_path / "report.json"
        mcr.export_json(str(json_file), include_comparisons=True)
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Should include pairwise comparisons or win probabilities
        assert 'pairwise_win_probabilities' in data or 'pairwise_comparisons' in data
    
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError):
            MultiCategoryRanker(method='invalid_method')
    
    def test_empty_category(self):
        """Test handling of empty category."""
        comparisons = {
            'cat1': [('A', 'B')],
            'cat2': [],  # Empty category
        }
        
        mcr = MultiCategoryRanker(method='bradley_terry')
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            mcr.fit(comparisons)


class TestRealisticTournament:
    """Realistic tournament scenario with 3 LLMs across multiple categories."""
    
    @pytest.fixture
    def llm_comparisons(self):
        """Realistic pairwise comparisons for 3 LLMs across multiple categories."""
        return {
            'temporal_grounding': [
                ('GPT-4V', 'Claude-3.5'),
                ('GPT-4V', 'Gemini'),
                ('Claude-3.5', 'Gemini'),
                ('GPT-4V', 'Claude-3.5'),  # Multiple comparisons for robustness
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
    
    def test_bradley_terry_tournament(self, llm_comparisons):
        """Test Bradley-Terry ranking on realistic LLM tournament."""
        # Test single category first
        bt = BradleyTerryRanker()
        bt.fit(llm_comparisons['temporal_grounding'])
        
        rankings = bt.get_rankings()
        rankings_dict = dict(rankings)
        
        # Verify all 3 LLMs are ranked
        assert len(rankings) == 3
        assert 'GPT-4V' in rankings_dict
        assert 'Claude-3.5' in rankings_dict
        assert 'Gemini' in rankings_dict
        
        # Verify scores are normalized
        scores = list(rankings_dict.values())
        assert np.isclose(sum(scores), 1.0)
        
        # GPT-4V should be ranked highest (wins most comparisons)
        assert rankings[0][0] == 'GPT-4V'
        assert rankings_dict['GPT-4V'] > rankings_dict['Claude-3.5']
        assert rankings_dict['GPT-4V'] > rankings_dict['Gemini']
        
        # Verify win probabilities
        prob_gpt_beats_claude = bt.get_win_probability('GPT-4V', 'Claude-3.5')
        prob_claude_beats_gpt = bt.get_win_probability('Claude-3.5', 'GPT-4V')
        assert prob_gpt_beats_claude > 0.5  # GPT-4V should be favored
        assert np.isclose(prob_gpt_beats_claude + prob_claude_beats_gpt, 1.0)
        
        # Test confidence intervals
        cis = bt.get_confidence_intervals(
            llm_comparisons['temporal_grounding'],
            n_bootstrap=100,
            confidence=0.95
        )
        assert 'GPT-4V' in cis
        assert 'Claude-3.5' in cis
        assert 'Gemini' in cis
        
        # CI should contain point estimate
        for model in ['GPT-4V', 'Claude-3.5', 'Gemini']:
            ci_lower, ci_upper = cis[model]
            assert ci_lower < ci_upper
            assert ci_lower <= rankings_dict[model] <= ci_upper
    
    def test_elo_tournament(self, llm_comparisons):
        """Test Elo ranking on realistic LLM tournament."""
        elo = EloRanker(k_factor=32, initial_rating=1500)
        
        # Record all matches sequentially
        for winner, loser in llm_comparisons['temporal_grounding']:
            elo.record_match(winner, loser)
        
        rankings = elo.get_rankings()
        ratings_dict = dict(rankings)
        
        # Verify all 3 LLMs are ranked
        assert len(rankings) == 3
        assert 'GPT-4V' in ratings_dict
        assert 'Claude-3.5' in ratings_dict
        assert 'Gemini' in ratings_dict
        
        # GPT-4V should have highest rating (wins most matches)
        assert rankings[0][0] == 'GPT-4V'
        assert ratings_dict['GPT-4V'] > ratings_dict['Claude-3.5']
        assert ratings_dict['GPT-4V'] > ratings_dict['Gemini']
        
        # Verify expected scores
        expected_gpt_vs_claude = elo.expected_score(
            ratings_dict['GPT-4V'],
            ratings_dict['Claude-3.5']
        )
        assert expected_gpt_vs_claude > 0.5  # GPT-4V should be favored
        
        # Verify history tracking
        history = elo.get_history()
        assert len(history) == len(llm_comparisons['temporal_grounding']) * 2  # 2 entries per match
        assert 'GPT-4V' in history['player'].values
        assert 'Claude-3.5' in history['player'].values
        assert 'Gemini' in history['player'].values
    
    def test_multicategory_bradley_terry(self, llm_comparisons):
        """Test multi-category ranking with Bradley-Terry method."""
        mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=100)
        mcr.fit(llm_comparisons)
        
        # Verify category rankings
        for category in llm_comparisons.keys():
            cat_rankings = mcr.get_category_rankings(category)
            cat_dict = dict(cat_rankings)
            
            # All categories should have all 3 LLMs
            assert len(cat_rankings) == 3
            assert 'GPT-4V' in cat_dict
            assert 'Claude-3.5' in cat_dict
            assert 'Gemini' in cat_dict
        
        # Verify overall rankings
        overall = mcr.get_overall_ranking()
        overall_dict = dict(overall)
        
        assert len(overall) == 3
        assert 'GPT-4V' in overall_dict
        assert 'Claude-3.5' in overall_dict
        assert 'Gemini' in overall_dict
        
        # GPT-4V should win overall (wins most categories)
        assert overall[0][0] == 'GPT-4V'
        assert overall_dict['GPT-4V'] > overall_dict['Claude-3.5']
        assert overall_dict['GPT-4V'] > overall_dict['Gemini']
        
        # Verify weighted aggregation
        # Give heavy weight to temporal_grounding where GPT-4V dominates
        weighted = mcr.get_overall_ranking(weights={
            'temporal_grounding': 0.5,
            'entity_recognition': 0.2,
            'clarification_questions': 0.15,
            'answer_quality': 0.15,
        })
        assert weighted[0][0] == 'GPT-4V'
    
    def test_multicategory_elo(self, llm_comparisons):
        """Test multi-category ranking with Elo method."""
        mcr = MultiCategoryRanker(method='elo', k_factor=32)
        mcr.fit(llm_comparisons)
        
        # Verify category rankings
        for category in llm_comparisons.keys():
            cat_rankings = mcr.get_category_rankings(category)
            cat_dict = dict(cat_rankings)
            
            assert len(cat_rankings) == 3
            assert 'GPT-4V' in cat_dict
            assert 'Claude-3.5' in cat_dict
            assert 'Gemini' in cat_dict
        
        # Verify overall rankings
        overall = mcr.get_overall_ranking()
        overall_dict = dict(overall)
        
        assert len(overall) == 3
        assert overall[0][0] == 'GPT-4V'  # Should win overall
        
        # Verify weighted aggregation
        weighted = mcr.get_overall_ranking(weights={
            'temporal_grounding': 0.4,
            'entity_recognition': 0.3,
            'clarification_questions': 0.15,
            'answer_quality': 0.15,
        })
        assert weighted[0][0] == 'GPT-4V'
    
    def test_export_functionality(self, llm_comparisons, tmp_path):
        """Test CSV and JSON export with realistic tournament data."""
        mcr = MultiCategoryRanker(method='bradley_terry', n_bootstrap=100)
        mcr.fit(llm_comparisons)
        
        # Test CSV export
        csv_file = tmp_path / "llm_rankings.csv"
        mcr.export_csv(str(csv_file))
        
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        
        # Verify columns
        assert 'model' in df.columns
        assert 'overall_score' in df.columns
        for category in llm_comparisons.keys():
            assert category in df.columns
        
        # Verify data
        assert len(df) == 3
        assert set(df['model'].values) == {'GPT-4V', 'Claude-3.5', 'Gemini'}
        
        # GPT-4V should have highest overall score
        gpt_row = df[df['model'] == 'GPT-4V'].iloc[0]
        assert gpt_row['overall_score'] > df[df['model'] == 'Claude-3.5'].iloc[0]['overall_score']
        assert gpt_row['overall_score'] > df[df['model'] == 'Gemini'].iloc[0]['overall_score']
        
        # Test JSON export
        json_file = tmp_path / "llm_report.json"
        mcr.export_json(str(json_file), include_comparisons=True)
        
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        
        # Verify structure
        assert data['method'] == 'bradley_terry'
        assert 'timestamp' in data
        assert set(data['categories']) == set(llm_comparisons.keys())
        
        # Verify overall rankings
        assert len(data['overall_rankings']) == 3
        overall_models = [entry['model'] for entry in data['overall_rankings']]
        assert set(overall_models) == {'GPT-4V', 'Claude-3.5', 'Gemini'}
        assert data['overall_rankings'][0]['model'] == 'GPT-4V'
        assert 'ci_lower' in data['overall_rankings'][0]
        assert 'ci_upper' in data['overall_rankings'][0]
        
        # Verify category rankings
        for category in llm_comparisons.keys():
            assert category in data['category_rankings']
            cat_rankings = data['category_rankings'][category]
            assert len(cat_rankings) == 3
            cat_models = [entry['model'] for entry in cat_rankings]
            assert set(cat_models) == {'GPT-4V', 'Claude-3.5', 'Gemini'}
        
        # Verify pairwise win probabilities
        assert 'pairwise_win_probabilities' in data
        for category in llm_comparisons.keys():
            assert category in data['pairwise_win_probabilities']
        
        # Verify metadata
        assert 'metadata' in data
        assert data['metadata']['n_models'] == 3
        assert len(data['metadata']['n_comparisons_per_category']) == 4
    
    def test_method_consistency(self, llm_comparisons):
        """Test that Bradley-Terry and Elo produce consistent rankings."""
        # Bradley-Terry
        mcr_bt = MultiCategoryRanker(method='bradley_terry')
        mcr_bt.fit(llm_comparisons)
        overall_bt = mcr_bt.get_overall_ranking()
        
        # Elo
        mcr_elo = MultiCategoryRanker(method='elo', k_factor=32)
        mcr_elo.fit(llm_comparisons)
        overall_elo = mcr_elo.get_overall_ranking()
        
        # Both should rank GPT-4V first
        assert overall_bt[0][0] == 'GPT-4V'
        assert overall_elo[0][0] == 'GPT-4V'
        
        # Both should have all 3 LLMs
        bt_models = [model for model, _ in overall_bt]
        elo_models = [model for model, _ in overall_elo]
        assert set(bt_models) == set(elo_models) == {'GPT-4V', 'Claude-3.5', 'Gemini'}
        
        # Verify category-level consistency for temporal_grounding
        bt_temporal = mcr_bt.get_category_rankings('temporal_grounding')
        elo_temporal = mcr_elo.get_category_rankings('temporal_grounding')
        
        # Both should rank GPT-4V first in temporal_grounding
        assert bt_temporal[0][0] == 'GPT-4V'
        assert elo_temporal[0][0] == 'GPT-4V'
