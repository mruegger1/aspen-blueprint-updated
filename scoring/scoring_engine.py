#!/usr/bin/env python3
"""
Comp Scoring Engine for Aspen Comp Finder
-----------------------------------------
Implements modular scoring logic for comparable properties.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os

# Set up logging
logger = logging.getLogger("aspen_comp_finder.scoring_engine")

class CompScoringEngine:
    """Scoring engine for comparable properties."""
    
    def __init__(self, weights=None, config_path=None):
        """
        Initialize the scoring engine.
        
        Args:
            weights (dict): Dictionary of scoring weights for different criteria
            config_path (str): Path to config file containing scoring weights
        """
        self.weights = weights or {}
        
        # If no weights provided, try to load from config
        if not self.weights and config_path:
            self._load_config(config_path)
        
        # If still no weights, use default hardcoded values
        if not self.weights:
            self._set_default_weights()
    
    def _set_default_weights(self):
        """Set default scoring weights."""
        self.weights = {
            'bedrooms': 2.0,
            'bathrooms': 2.0,
            'property_type': 2.0,
            'area': 1.5,
            'condition': 1.5,
            'sqft': 1.0,
            'str': 1.0,
            'recency': 1.0,
            'building_premium': 1.2,
            'street_premium': 0.8
        }
    
    def _load_config(self, config_path):
        """
        Load scoring weights from a config file.
        
        Args:
            config_path (str): Path to JSON config file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'scoring_weights' in config:
                        self.weights = config['scoring_weights']
                        logger.info(f"Loaded scoring weights from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading scoring weights from {config_path}: {str(e)}")
            # Fall back to default weights
            self._set_default_weights()
    
    def score(self, subject, comps, custom_weights=None):
        """
        Score comps based on similarity to subject property using weighted criteria.
        
        Args:
            subject (Series): Subject property
            comps (DataFrame): Potential comparable properties
            custom_weights (dict): Optional custom scoring weights to override defaults
            
        Returns:
            DataFrame: Comps with match scores
        """
        # Use custom weights if provided, otherwise use instance weights
        weights = custom_weights or self.weights
        
        # Create a copy of comps to work with
        df = comps.copy()
        df['match_score'] = 0.0
        
        # Apply individual scoring criteria
        df = self._score_bedrooms(df, subject, weights)
        df = self._score_bathrooms(df, subject, weights)
        df = self._score_property_type(df, subject, weights)
        df = self._score_area(df, subject, weights)
        df = self._score_condition(df, subject, weights)
        df = self._score_sqft(df, subject, weights)
        df = self._score_str_eligibility(df, subject, weights)
        df = self._score_recency(df, weights)
        df = self._score_building_premium(df, subject, weights)
        df = self._score_street_premium(df, subject, weights)
        
        # Normalize match score
        total_weight = sum(weights.values())
        df['match_score'] = (df['match_score'] / total_weight) * 100
        
        return df
    
    def _score_bedrooms(self, df, subject, weights):
        """Score properties based on bedroom match."""
        if 'bedrooms' in subject and 'bedrooms' in df.columns:
            bedrooms = subject['bedrooms']
            df['bedroom_score'] = 1.0 - (0.25 * np.abs(df['bedrooms'] - bedrooms))
            df['bedroom_score'] = df['bedroom_score'].clip(0, 1)
            df['match_score'] += weights.get('bedrooms', 0) * df['bedroom_score']
        
        return df
    
    def _score_bathrooms(self, df, subject, weights):
        """Score properties based on bathroom match."""
        if 'total_baths' in subject and 'total_baths' in df.columns:
            bathrooms = subject['total_baths']
            df['bathroom_score'] = 1.0 - (0.25 * np.abs(df['total_baths'] - bathrooms))
            df['bathroom_score'] = df['bathroom_score'].clip(0, 1)
            df['match_score'] += weights.get('bathrooms', 0) * df['bathroom_score']
        
        return df
    
    def _score_property_type(self, df, subject, weights):
        """Score properties based on property type match."""
        if 'resolved_property_type' in subject and 'resolved_property_type' in df.columns:
            property_type = subject['resolved_property_type']
            
            # Property type map
            property_type_map = {
                'Condo': ['Condo', 'Condominium', 'Apartment'],
                'Townhouse': ['Townhouse', 'Townhome', 'Town House', 'Townhome/Condo'],
                'Single Family': ['Single Family', 'House', 'Detached', 'Single-Family']
            }
            
            # Normalize property type
            normalized_type = None
            for category, variants in property_type_map.items():
                if property_type and any(prop_variant.lower() in property_type.lower() for prop_variant in variants):
                    normalized_type = category
                    break
            
            if normalized_type:
                df['property_type_score'] = 0.0
                for category, variants in property_type_map.items():
                    match_score = 1.0 if category == normalized_type else 0.3
                    # Check if each property's type is in the variants list
                    for idx, row in df.iterrows():
                        if row.get('resolved_property_type') and any(
                            prop_variant.lower() in row['resolved_property_type'].lower() 
                            for prop_variant in variants
                        ):
                            df.at[idx, 'property_type_score'] = match_score
            else:
                # Direct comparison
                df['property_type_score'] = np.where(
                    df['resolved_property_type'].str.lower() == (property_type.lower() if property_type else ''), 
                    1.0, 
                    0.0
                )
            
            df['match_score'] += weights.get('property_type', 0) * df['property_type_score']
        
        return df
    
    def _score_area(self, df, subject, weights):
        """Score properties based on area/neighborhood match."""
        if 'area' in subject and 'area' in df.columns:
            area = subject['area']
            
            # Area proximity map
            area_proximity = {
                'Core': ['Core', 'Downtown', 'Central Core'],
                'West End': ['West End', 'West Side'],
                'East End': ['East End', 'East Side'],
                'Red Mountain': ['Red Mountain', 'Red Mtn'],
                'Smuggler': ['Smuggler', 'Smuggler Mountain'],
                'McLain Flats': ['McLain Flats', 'McLain'],
                'Woody Creek': ['Woody Creek'],
                'Starwood': ['Starwood']
            }
            
            # Adjacent areas
            adjacent_areas = {
                'Core': ['West End', 'East End'],
                'West End': ['Core', 'Red Mountain'],
                'East End': ['Core', 'Smuggler'],
                'Red Mountain': ['West End'],
                'Smuggler': ['East End']
            }
            
            # Normalize area
            normalized_area = None
            for category, variants in area_proximity.items():
                if area and any(area_variant.lower() in area.lower() for area_variant in variants):
                    normalized_area = category
                    break
            
            if normalized_area:
                df['area_score'] = 0.0
                for category, variants in area_proximity.items():
                    # Exact match gets 1.0
                    match_score = 1.0 if category == normalized_area else 0.0
                    
                    # Adjacent areas get 0.7
                    if category != normalized_area and normalized_area in adjacent_areas and category in adjacent_areas[normalized_area]:
                        match_score = 0.7
                    
                    # Apply score to properties in this area
                    for idx, row in df.iterrows():
                        if row.get('area') and any(
                            area_variant.lower() in row['area'].lower() 
                            for area_variant in variants
                        ):
                            df.at[idx, 'area_score'] = match_score
            else:
                # Direct comparison
                df['area_score'] = np.where(
                    df['area'].str.lower() == (area.lower() if area else ''), 
                    1.0, 
                    0.0
                )
            
            df['match_score'] += weights.get('area', 0) * df['area_score']
        
        return df
    
    def _score_condition(self, df, subject, weights):
        """Score properties based on condition match."""
        if 'improved_condition' in subject and 'improved_condition' in df.columns:
            condition = subject['improved_condition']
            condition_values = ['Excellent', 'Good', 'Average', 'Fair', 'Poor']
            
            # Find index of target condition
            if condition in condition_values:
                target_idx = condition_values.index(condition)
                
                # Score based on distance in condition scale
                df['condition_score'] = 0.0
                for idx, row in df.iterrows():
                    if pd.isna(row['improved_condition']) or row['improved_condition'] not in condition_values:
                        df.at[idx, 'condition_score'] = 0.5  # Default for unknown
                    else:
                        prop_idx = condition_values.index(row['improved_condition'])
                        distance = abs(target_idx - prop_idx)
                        
                        # Score decreases with distance
                        if distance == 0:
                            df.at[idx, 'condition_score'] = 1.0
                        elif distance == 1:
                            df.at[idx, 'condition_score'] = 0.8
                        else:
                            df.at[idx, 'condition_score'] = max(0, 1.0 - (distance * 0.3))
            else:
                # Direct comparison
                df['condition_score'] = np.where(
                    df['improved_condition'] == condition,
                    1.0,
                    0.5
                )
            
            df['match_score'] += weights.get('condition', 0) * df['condition_score']
        
        return df
    
    def _score_sqft(self, df, subject, weights):
        """Score properties based on square footage match."""
        if 'total_sqft' in subject and 'total_sqft' in df.columns:
            target_sqft = subject['total_sqft']
            
            # Score based on percentage difference
            df['sqft_score'] = 0.0
            for idx, row in df.iterrows():
                if pd.isna(row['total_sqft']) or row['total_sqft'] == 0:
                    df.at[idx, 'sqft_score'] = 0.5  # Default for missing data
                else:
                    pct_diff = abs(row['total_sqft'] - target_sqft) / target_sqft if target_sqft else 1.0
                    
                    # Higher score for closer match
                    if pct_diff <= 0.05:  # Within 5%
                        df.at[idx, 'sqft_score'] = 1.0
                    elif pct_diff <= 0.10:  # Within 10%
                        df.at[idx, 'sqft_score'] = 0.9
                    elif pct_diff <= 0.20:  # Within 20%
                        df.at[idx, 'sqft_score'] = 0.8
                    elif pct_diff <= 0.30:  # Within 30%
                        df.at[idx, 'sqft_score'] = 0.6
                    else:
                        df.at[idx, 'sqft_score'] = max(0, 1.0 - pct_diff)
            
            df['match_score'] += weights.get('sqft', 0) * df['sqft_score']
        
        return df
    
    def _score_str_eligibility(self, df, subject, weights):
        """Score properties based on short-term rental eligibility match."""
        if 'str_eligible' in subject and 'str_eligible' in df.columns:
            str_eligible = subject['str_eligible']
            # Standardize format
            if isinstance(str_eligible, str):
                str_eligible = str_eligible.lower() in ['yes', 'true', 'y', '1']
            
            df['str_score'] = 0.0
            for idx, row in df.iterrows():
                row_str = row['str_eligible']
                if isinstance(row_str, str):
                    row_str = row_str.lower() in ['yes', 'true', 'y', '1']
                
                df.at[idx, 'str_score'] = 1.0 if row_str == str_eligible else 0.0
            
            df['match_score'] += weights.get('str', 0) * df['str_score']
        
        return df
    
    def _score_recency(self, df, weights):
        """Score properties based on recency of sale."""
        if 'sold_date_dt' in df.columns:
            # More recent sales get higher scores
            now = datetime.now()
            df['days_since_sale'] = (now - df['sold_date_dt']).dt.days
            
            # Normalize to 0-1 range
            max_days = df['days_since_sale'].max()
            min_days = df['days_since_sale'].min()
            
            if max_days > min_days:
                df['recency_score'] = 1.0 - ((df['days_since_sale'] - min_days) / (max_days - min_days))
            else:
                df['recency_score'] = 1.0
            
            df['match_score'] += weights.get('recency', 0) * df['recency_score']
        
        return df
    
    def _score_building_premium(self, df, subject, weights):
        """Score properties based on building premium match."""
        if 'building_premium' in subject and 'building_premium' in df.columns:
            subject_premium = subject['building_premium']
            
            # Score based on premium difference
            df['building_premium_score'] = 0.0
            for idx, row in df.iterrows():
                premium_diff = abs(row['building_premium'] - subject_premium)
                
                # Closer premium gets higher score
                if premium_diff <= 0.05:
                    df.at[idx, 'building_premium_score'] = 1.0
                elif premium_diff <= 0.1:
                    df.at[idx, 'building_premium_score'] = 0.9
                elif premium_diff <= 0.2:
                    df.at[idx, 'building_premium_score'] = 0.8
                elif premium_diff <= 0.3:
                    df.at[idx, 'building_premium_score'] = 0.6
                else:
                    df.at[idx, 'building_premium_score'] = max(0, 1.0 - premium_diff)
            
            df['match_score'] += weights.get('building_premium', 0) * df['building_premium_score']
        
        return df
    
    def _score_street_premium(self, df, subject, weights):
        """Score properties based on street premium match."""
        if 'street_premium' in subject and 'street_premium' in df.columns:
            subject_premium = subject['street_premium']
            
            # Score based on premium difference
            df['street_premium_score'] = 0.0
            for idx, row in df.iterrows():
                premium_diff = abs(row['street_premium'] - subject_premium)
                
                # Closer premium gets higher score
                if premium_diff <= 0.05:
                    df.at[idx, 'street_premium_score'] = 1.0
                elif premium_diff <= 0.1:
                    df.at[idx, 'street_premium_score'] = 0.9
                elif premium_diff <= 0.2:
                    df.at[idx, 'street_premium_score'] = 0.7
                else:
                    df.at[idx, 'street_premium_score'] = max(0, 1.0 - premium_diff)
            
            df['match_score'] += weights.get('street_premium', 0) * df['street_premium_score']
        
        return df