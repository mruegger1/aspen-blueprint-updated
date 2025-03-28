#!/usr/bin/env python3
"""
Unit tests for CompScoringEngine
--------------------------------
Tests the scoring logic for comparable properties.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from aspen_comp_finder.scoring.scoring_engine import CompScoringEngine


class TestCompScoringEngine(unittest.TestCase):
    """Test cases for the CompScoringEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test subject property
        self.subject = pd.Series({
            'bedrooms': 2,
            'total_baths': 2.0,
            'resolved_property_type': 'Condo',
            'area': 'Core',
            'improved_condition': 'Good',
            'total_sqft': 1200,
            'str_eligible': True,
            'building_premium': 0.1,
            'street_premium': 0.05
        })
        
        # Create test comp properties
        self.comps = pd.DataFrame([
            # Perfect match
            {
                'bedrooms': 2,
                'total_baths': 2.0,
                'resolved_property_type': 'Condo',
                'area': 'Core',
                'improved_condition': 'Good',
                'total_sqft': 1200,
                'str_eligible': 'Yes',
                'sold_date_dt': datetime.now() - timedelta(days=30),
                'building_premium': 0.1,
                'street_premium': 0.05
            },
            # Close match
            {
                'bedrooms': 2,
                'total_baths': 2.5,
                'resolved_property_type': 'Condo',
                'area': 'Core',
                'improved_condition': 'Excellent',
                'total_sqft': 1300,
                'str_eligible': 'Yes',
                'sold_date_dt': datetime.now() - timedelta(days=60),
                'building_premium': 0.12,
                'street_premium': 0.04
            },
            # Poor match
            {
                'bedrooms': 3,
                'total_baths': 3.0,
                'resolved_property_type': 'Townhouse',
                'area': 'West End',
                'improved_condition': 'Fair',
                'total_sqft': 1800,
                'str_eligible': 'No',
                'sold_date_dt': datetime.now() - timedelta(days=365),
                'building_premium': 0.3,
                'street_premium': 0.15
            }
        ])
        
        # Create scoring engine with default weights
        self.scoring_engine = CompScoringEngine()
    
    def test_overall_scoring(self):
        """Test that overall scoring ranks properties correctly."""
        # Score the comps
        scored_comps = self.scoring_engine.score(self.subject, self.comps)
        
        # Check if the perfect match is ranked highest
        self.assertEqual(scored_comps['match_score'].idxmax(), 0)
        
        # Check if the poor match is ranked lowest
        self.assertEqual(scored_comps['match_score'].idxmin(), 2)
        
        # Verify score ranges
        self.assertGreater(scored_comps.iloc[0]['match_score'], 90)  # Perfect match > 90%
        self.assertLess(scored_comps.iloc[2]['match_score'], 60)     # Poor match < 60%
    
    def test_bedroom_scoring(self):
        """Test bedroom scoring logic."""
        # Set up test data
        test_subject = pd.Series({'bedrooms': 2})
        test_comps = pd.DataFrame({
            'bedrooms': [2, 1, 3, 4]
        })
        
        # Score only bedrooms
        test_weights = {'bedrooms': 1.0}
        scored = self.scoring_engine._score_bedrooms(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['bedroom_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['bedroom_score'], 0.75)  # Off by 1
        self.assertEqual(scored.iloc[2]['bedroom_score'], 0.75)  # Off by 1
        self.assertEqual(scored.iloc[3]['bedroom_score'], 0.5)   # Off by 2
    
    def test_bathroom_scoring(self):
        """Test bathroom scoring logic."""
        # Set up test data
        test_subject = pd.Series({'total_baths': 2.5})
        test_comps = pd.DataFrame({
            'total_baths': [2.5, 2.0, 3.0, 4.0]
        })
        
        # Score only bathrooms
        test_weights = {'bathrooms': 1.0}
        scored = self.scoring_engine._score_bathrooms(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['bathroom_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['bathroom_score'], 0.875)  # Off by 0.5
        self.assertEqual(scored.iloc[2]['bathroom_score'], 0.875)  # Off by 0.5
        self.assertEqual(scored.iloc[3]['bathroom_score'], 0.625)  # Off by 1.5
    
    def test_property_type_scoring(self):
        """Test property type scoring logic."""
        # Set up test data
        test_subject = pd.Series({'resolved_property_type': 'Condo'})
        test_comps = pd.DataFrame({
            'resolved_property_type': ['Condo', 'Condominium', 'Townhouse', 'Single Family']
        })
        
        # Score only property type
        test_weights = {'property_type': 1.0}
        scored = self.scoring_engine._score_property_type(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['property_type_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['property_type_score'], 1.0)  # Same category
        self.assertLess(scored.iloc[2]['property_type_score'], 0.5)   # Different category
    
    def test_area_scoring(self):
        """Test area/neighborhood scoring logic."""
        # Set up test data
        test_subject = pd.Series({'area': 'Core'})
        test_comps = pd.DataFrame({
            'area': ['Core', 'West End', 'East End', 'Red Mountain']
        })
        
        # Score only area
        test_weights = {'area': 1.0}
        scored = self.scoring_engine._score_area(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['area_score'], 1.0)  # Exact match
        self.assertGreater(scored.iloc[1]['area_score'], 0.5)  # Adjacent area
        self.assertGreater(scored.iloc[2]['area_score'], 0.5)  # Adjacent area
        self.assertEqual(scored.iloc[3]['area_score'], 0.0)  # Non-adjacent area
    
    def test_condition_scoring(self):
        """Test property condition scoring logic."""
        # Set up test data
        test_subject = pd.Series({'improved_condition': 'Good'})
        test_comps = pd.DataFrame({
            'improved_condition': ['Excellent', 'Good', 'Average', 'Poor']
        })
        
        # Score only condition
        test_weights = {'condition': 1.0}
        scored = self.scoring_engine._score_condition(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['condition_score'], 0.8)  # One level above
        self.assertEqual(scored.iloc[1]['condition_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[2]['condition_score'], 0.8)  # One level below
        self.assertLess(scored.iloc[3]['condition_score'], 0.5)   # Two levels below
    
    def test_sqft_scoring(self):
        """Test square footage scoring logic."""
        # Set up test data
        test_subject = pd.Series({'total_sqft': 1000})
        test_comps = pd.DataFrame({
            'total_sqft': [1000, 1050, 1200, 1500]
        })
        
        # Score only sqft
        test_weights = {'sqft': 1.0}
        scored = self.scoring_engine._score_sqft(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['sqft_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['sqft_score'], 1.0)  # Within 5%
        self.assertEqual(scored.iloc[2]['sqft_score'], 0.8)  # Within 20%
        self.assertLess(scored.iloc[3]['sqft_score'], 0.7)   # More than 20% difference
    
    def test_str_eligibility_scoring(self):
        """Test STR eligibility scoring logic."""
        # Set up test data
        test_subject = pd.Series({'str_eligible': True})
        test_comps = pd.DataFrame({
            'str_eligible': ['Yes', 'No', 'True', 'False']
        })
        
        # Score only STR
        test_weights = {'str': 1.0}
        scored = self.scoring_engine._score_str_eligibility(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['str_score'], 1.0)  # Yes matches True
        self.assertEqual(scored.iloc[1]['str_score'], 0.0)  # No doesn't match True
        self.assertEqual(scored.iloc[2]['str_score'], 1.0)  # True matches True
        self.assertEqual(scored.iloc[3]['str_score'], 0.0)  # False doesn't match True
    
    def test_recency_scoring(self):
        """Test temporal/recency scoring logic."""
        # Set up test data with different sale dates
        now = datetime.now()
        test_comps = pd.DataFrame({
            'sold_date_dt': [
                now - timedelta(days=30),  # 1 month ago
                now - timedelta(days=90),  # 3 months ago
                now - timedelta(days=180), # 6 months ago
                now - timedelta(days=365)  # 1 year ago
            ]
        })
        
        # Score only recency
        test_weights = {'recency': 1.0}
        scored = self.scoring_engine._score_recency(test_comps, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['recency_score'], 1.0)  # Most recent
        self.assertGreater(scored.iloc[1]['recency_score'], scored.iloc[2]['recency_score'])  # Newer > older
        self.assertGreater(scored.iloc[2]['recency_score'], scored.iloc[3]['recency_score'])  # Newer > older
        self.assertEqual(scored.iloc[3]['recency_score'], 0.0)  # Oldest
    
    def test_building_premium_scoring(self):
        """Test building premium scoring logic."""
        # Set up test data
        test_subject = pd.Series({'building_premium': 0.1})
        test_comps = pd.DataFrame({
            'building_premium': [0.1, 0.12, 0.2, 0.4]
        })
        
        # Score only building premium
        test_weights = {'building_premium': 1.0}
        scored = self.scoring_engine._score_building_premium(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['building_premium_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['building_premium_score'], 0.9)  # Within 0.05
        self.assertEqual(scored.iloc[2]['building_premium_score'], 0.8)  # Within 0.1
        self.assertLess(scored.iloc[3]['building_premium_score'], 0.7)   # More than 0.2 difference
    
    def test_street_premium_scoring(self):
        """Test street premium scoring logic."""
        # Set up test data
        test_subject = pd.Series({'street_premium': 0.05})
        test_comps = pd.DataFrame({
            'street_premium': [0.05, 0.08, 0.2, -0.1]
        })
        
        # Score only street premium
        test_weights = {'street_premium': 1.0}
        scored = self.scoring_engine._score_street_premium(test_comps, test_subject, test_weights)
        
        # Check scores
        self.assertEqual(scored.iloc[0]['street_premium_score'], 1.0)  # Exact match
        self.assertEqual(scored.iloc[1]['street_premium_score'], 0.9)  # Within 0.05
        self.assertEqual(scored.iloc[2]['street_premium_score'], 0.7)  # Within 0.2
        self.assertLess(scored.iloc[3]['street_premium_score'], 0.7)   # More than 0.1 difference
    
    def test_custom_weights(self):
        """Test scoring with custom weights."""
        # Standard weights
        standard_scored = self.scoring_engine.score(self.subject, self.comps)
        
        # Custom weights prioritizing bedrooms and bathrooms
        custom_weights = {
            'bedrooms': 5.0,      # Much higher than default
            'bathrooms': 5.0,     # Much higher than default
            'property_type': 1.0, # Lower than default
            'area': 1.0,          # Lower than default
            'condition': 1.0,     # Lower than default
            'sqft': 0.5,          # Lower than default
            'str': 0.5,           # Lower than default
            'recency': 0.5,       # Lower than default
            'building_premium': 0.5,  # Lower than default
            'street_premium': 0.5     # Lower than default
        }
        
        custom_scored = self.scoring_engine.score(self.subject, self.comps, custom_weights)
        
        # The second comp has different bathrooms, so it should be more penalized
        # with the custom weights than with standard weights
        std_diff = standard_scored.iloc[0]['match_score'] - standard_scored.iloc[1]['match_score']
        custom_diff = custom_scored.iloc[0]['match_score'] - custom_scored.iloc[1]['match_score']
        
        # The difference should be greater with custom weights
        self.assertGreater(custom_diff, std_diff)


if __name__ == '__main__':
    unittest.main()