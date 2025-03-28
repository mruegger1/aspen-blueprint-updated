#!/usr/bin/env python3
"""
Property Filters for Aspen Comp Finder
-------------------------------------
Implements filtering logic with progressive relaxation capabilities.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logger = logging.getLogger("aspen_comp_finder.filters")

class PropertyFilter:
    """Handles filtering of property data with progressive relaxation."""
    
    def __init__(self):
        """Initialize the property filter."""
        pass
    
    def filter_basic_criteria(self, df, criteria):
        """
        Apply basic filters to property data.
        
        Args:
            df (DataFrame): Property data
            criteria (dict): Filter criteria
            
        Returns:
            DataFrame: Filtered data
        """
        # Make a copy of the data
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Apply filters
        if criteria.get('bedrooms') is not None:
            filtered_df = filtered_df[filtered_df['bedrooms'] == criteria['bedrooms']]
        
        if criteria.get('bathrooms') is not None:
            filtered_df = filtered_df[filtered_df['total_baths'] == criteria['bathrooms']]
        
        if criteria.get('property_type') is not None and 'resolved_property_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['resolved_property_type'] == criteria['property_type']]
        
        if criteria.get('area') is not None and 'area' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['area'] == criteria['area']]
        
        if criteria.get('str_eligible') is not None and 'str_eligible' in filtered_df.columns:
            # Handle different formats of str_eligible
            str_value = criteria['str_eligible']
            if isinstance(str_value, bool):
                str_value = 'Yes' if str_value else 'No'
            filtered_df = filtered_df[filtered_df['str_eligible'].astype(str).str.lower() == str_value.lower()]
        
        if criteria.get('max_price') is not None:
            price_col = 'adjusted_sold_price_time' if 'adjusted_sold_price_time' in filtered_df.columns else 'adjusted_sold_price'
            filtered_df = filtered_df[filtered_df[price_col] <= criteria['max_price']]
        
        if criteria.get('sqft_min') is not None and 'total_sqft' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['total_sqft'] >= criteria['sqft_min']]
        
        if criteria.get('sqft_max') is not None and 'total_sqft' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['total_sqft'] <= criteria['sqft_max']]
        
        if criteria.get('year_built_min') is not None and 'year_built' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['year_built'] >= criteria['year_built_min']]
        
        # Filter by listing status
        if criteria.get('listing_status') is not None and 'listing_status' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['listing_status'] == criteria['listing_status']]
        
        # Filter for recent sales (only for sold properties)
        if criteria.get('listing_status') in [None, 'S', 'Sold', 'SOLD'] and 'sold_date_dt' in filtered_df.columns:
            months_back = criteria.get('months_back', 24)
            cutoff_date = datetime.now() - timedelta(days=30 * months_back)
            filtered_df = filtered_df[filtered_df['sold_date_dt'] >= cutoff_date]
        
        logger.info(f"Applied basic filters: {initial_count} -> {len(filtered_df)} properties")
        
        # Exclude subject property if provided
        filtered_df = self._exclude_subject_property(filtered_df, criteria)
        
        return filtered_df
    
    def _exclude_subject_property(self, df, criteria):
        """
        Exclude the subject property from results.
        
        Args:
            df (DataFrame): Property data
            criteria (dict): Filter criteria containing exclusion info
            
        Returns:
            DataFrame: Data with subject property excluded
        """
        if criteria.get('exclude_address'):
            exclude_address = criteria['exclude_address']
            address_col = None
            
            # Find address column
            for col in ['full_address', 'address', 'street_address', 'location']:
                if col in df.columns:
                    address_col = col
                    break
            
            if address_col:
                # Count before
                before_count = len(df)
                
                # Exclude the subject property
                df = df[~df[address_col].str.contains(exclude_address, case=False, na=False)]
                
                # Log if found and excluded
                if len(df) < before_count:
                    logger.info(f"Excluded subject property: {exclude_address}")
            
            # Also try excluding by list_number if present
            if 'list_number' in df.columns and criteria.get('exclude_list_number'):
                df = df[df['list_number'] != criteria['exclude_list_number']]
        
        return df
    
    def apply_progressive_relaxation(self, df, criteria, min_comps=3):
        """
        Progressively relax filter criteria to find minimum number of comps.
        
        Args:
            df (DataFrame): Original property data
            criteria (dict): Original filter criteria
            min_comps (int): Minimum number of comps to find
            
        Returns:
            tuple: (DataFrame of filtered properties, list of relaxation steps applied)
        """
        # Apply basic filters
        filtered_df = self.filter_basic_criteria(df, criteria)
        
        # If enough comps, return early
        if len(filtered_df) >= min_comps:
            return filtered_df, []
        
        # Stage 1-N: Apply progressive relaxation
        relaxation_steps = []
        
        # Make a copy of criteria for relaxation
        relaxed_criteria = criteria.copy()
        
        # Stage 1: Relax price range by 25%
        if len(filtered_df) < min_comps and relaxed_criteria.get('max_price') is not None:
            relaxed_criteria['max_price'] = relaxed_criteria['max_price'] * 1.25
            relaxed_df = self.filter_basic_criteria(df, relaxed_criteria)
            
            if len(relaxed_df) >= min_comps:
                logger.info(f"Found {len(relaxed_df)} comps after relaxing price range by 25%")
                filtered_df = relaxed_df
                relaxation_steps.append("price_range_25pct")
        
        # Stage 2: Relax square footage range by 25%
        if len(filtered_df) < min_comps and (relaxed_criteria.get('sqft_min') is not None or relaxed_criteria.get('sqft_max') is not None):
            if relaxed_criteria.get('sqft_min') is not None:
                relaxed_criteria['sqft_min'] = relaxed_criteria['sqft_min'] * 0.75
            
            if relaxed_criteria.get('sqft_max') is not None:
                relaxed_criteria['sqft_max'] = relaxed_criteria['sqft_max'] * 1.25
            
            relaxed_df = self.filter_basic_criteria(df, relaxed_criteria)
            
            if len(relaxed_df) >= min_comps:
                logger.info(f"Found {len(relaxed_df)} comps after relaxing sqft range by 25%")
                filtered_df = relaxed_df
                relaxation_steps.append("sqft_range_25pct")
        
        # Stage 3: Increase months_back by 50%
        if len(filtered_df) < min_comps and relaxed_criteria.get('months_back') is not None:
            relaxed_criteria['months_back'] = int(relaxed_criteria['months_back'] * 1.5)
            relaxed_df = self.filter_basic_criteria(df, relaxed_criteria)
            
            if len(relaxed_df) >= min_comps:
                logger.info(f"Found {len(relaxed_df)} comps after extending time range to {relaxed_criteria['months_back']} months")
                filtered_df = relaxed_df
                relaxation_steps.append("time_range_extended")
        
        # Stage 4: Remove property type restriction
        if len(filtered_df) < min_comps and relaxed_criteria.get('property_type') is not None:
            relaxed_criteria['property_type'] = None
            relaxed_df = self.filter_basic_criteria(df, relaxed_criteria)
            
            if len(relaxed_df) >= min_comps:
                logger.info(f"Found {len(relaxed_df)} comps after removing property type filter")
                filtered_df = relaxed_df
                relaxation_steps.append("property_type_removed")
        
        # Stage 5: Remove area restriction as a last resort
        if len(filtered_df) < min_comps and relaxed_criteria.get('area') is not None:
            relaxed_criteria['area'] = None
            relaxed_df = self.filter_basic_criteria(df, relaxed_criteria)
            
            if len(relaxed_df) >= min_comps:
                logger.info(f"Found {len(relaxed_df)} comps after removing area filter")
                filtered_df = relaxed_df
                relaxation_steps.append("area_removed")
        
        # Log relaxation steps
        if relaxation_steps:
            logger.info(f"Applied filter relaxation: {', '.join(relaxation_steps)}")
        
        return filtered_df, relaxation_steps