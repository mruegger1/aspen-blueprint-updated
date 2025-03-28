"""
Core comp finding functionality for real estate analysis.
"""

import pandas as pd
import numpy as np
import re
import datetime

from utils import logger, clean_filename
from data_loader import DataLoader
from price_adjuster import PriceAdjuster
from address_matcher import AddressMatcher
from analysis import PropertyAnalyzer


class EnhancedCompFinder:
    """Finds comparable properties based on user-defined criteria using enriched dataset."""

    def __init__(self, csv_path=None, time_adjust=True):
        """
        Initialize the CompFinder with the path to the enriched real estate data.

        Args:
            csv_path (str, optional): Path to the CSV file containing real estate data.
                If None, will attempt to find the file in common locations.
            time_adjust (bool): Whether to apply time-based price adjustments.
        """
        self.time_adjust = time_adjust

        # Load and process data
        self.data, self.file_path = DataLoader.find_data_file(csv_path)
        self.data = DataLoader.preprocess_data(self.data)
        self.data = DataLoader.process_dates(self.data)

        # Add time-adjusted prices if requested
        if self.time_adjust:
            self.data = PriceAdjuster.add_time_adjusted_prices(self.data)

        # Check for enriched dataset features
        self.has_demo_score = "demo_score" in self.data.columns
        self.has_condition_data = "improved_condition" in self.data.columns
        self.has_tdr_data = "tdr_eligible_flag_lookup" in self.data.columns

    def find_comps(
        self,
        criteria=None,
        subject_property=None,
        min_comps=3,
        limit=10,
        sort_by=None,
        ascending=True,
    ):
        """
        Find comparable properties based on the given criteria, with automatic
        relaxation to ensure a minimum number of comparable properties.

        Args:
            criteria (dict): Dictionary of field:value pairs to filter properties by.
            subject_property: Property to exclude from results (if provided)
            min_comps (int): Minimum number of comps to try to find
            limit (int): Maximum number of comps to return
            sort_by (str): Field to sort results by
            ascending (bool): Sort direction

        Returns:
            tuple: (DataFrame of comps, list of relaxed filters)
        """
        if criteria is None:
            criteria = {}

        # Set default sort field based on time adjustment
        if sort_by is None:
            sort_by = (
                "adjusted_price_per_sqft_time" if self.time_adjust else "price_per_sqft"
            )

        # Keep track of which filters were relaxed
        relaxed_filters = []

        # Copy the initial criteria to preserve the original values
        original_criteria = criteria.copy()

        # Define relaxation stages
        relaxation_stages = [
            # Stage 1: Original criteria
            {},
            # Stage 2: Widen numeric ranges by 25%
            {"relax_numeric": 0.25},
            # Stage 3: Widen numeric ranges by 50% and relax condition
            {"relax_numeric": 0.50, "relax_condition": True},
            # Stage 4: Widen numeric ranges by 100% and relax condition & STR
            {"relax_numeric": 1.0, "relax_condition": True, "relax_str": True},
            # Stage 5: Minimal criteria (extreme relaxation)
            {
                "relax_numeric": 2.0,
                "relax_condition": True,
                "relax_str": True,
                "relax_property_type": True,
            },
        ]

        # Try each relaxation stage until we find enough comps
        for stage_idx, relaxations in enumerate(relaxation_stages):
            # Apply the current stage's relaxations to the criteria
            current_criteria = self._relax_criteria(original_criteria, relaxations)

            # If this isn't the first stage, log the relaxation
            if stage_idx > 0:
                relaxed_filters.extend(self._get_relaxed_filter_names(relaxations))
                if relaxations:
                    logger.info(
                        f"Relaxation stage {stage_idx}: {', '.join(self._get_relaxed_filter_names(relaxations))}"
                    )

            # Start with all data
            filtered_data = self.data.copy()

            # Apply filters based on criteria
            for field, value in current_criteria.items():
                # Skip if field not in data
                if field not in filtered_data.columns and not field.startswith(
                    ("min_", "max_")
                ):
                    continue

                # Basic property criteria
                if field == "min_bedrooms":
                    filtered_data = filtered_data[filtered_data["bedrooms"] >= value]
                elif field == "max_bedrooms":
                    filtered_data = filtered_data[filtered_data["bedrooms"] <= value]
                elif field == "min_baths":
                    filtered_data = filtered_data[filtered_data["total_baths"] >= value]
                elif field == "max_baths":
                    filtered_data = filtered_data[filtered_data["total_baths"] <= value]
                elif field == "min_sqft":
                    filtered_data = filtered_data[filtered_data["total_sqft"] >= value]
                elif field == "max_sqft":
                    filtered_data = filtered_data[filtered_data["total_sqft"] <= value]
                elif field == "max_walk_to_gondola":
                    filtered_data = filtered_data[
                        filtered_data["walk_time_to_gondola_min"] <= value
                    ]
                elif field == "property_type":
                    filtered_data = filtered_data[
                        filtered_data["property_type"].str.contains(
                            value, case=False, na=False
                        )
                    ]
                elif field == "area":
                    filtered_data = filtered_data[
                        filtered_data["area"].str.contains(value, case=False, na=False)
                    ]
                elif field == "str_eligible":
                    filtered_data = filtered_data[
                        filtered_data["str_eligible"] == value
                    ]
                elif field == "min_year":
                    filtered_data = filtered_data[
                        filtered_data["transaction_year"] >= value
                    ]
                elif field == "max_year":
                    filtered_data = filtered_data[
                        filtered_data["transaction_year"] <= value
                    ]
                # Enhanced criteria - demo score and condition
                elif (
                    field == "min_demo_score" and "demo_score" in filtered_data.columns
                ):
                    filtered_data = filtered_data[filtered_data["demo_score"] >= value]
                elif (
                    field == "max_demo_score" and "demo_score" in filtered_data.columns
                ):
                    filtered_data = filtered_data[filtered_data["demo_score"] <= value]
                elif (
                    field == "value_add_type"
                    and "value_add_type" in filtered_data.columns
                ):
                    filtered_data = filtered_data[
                        filtered_data["value_add_type"] == value
                    ]
                elif (
                    field == "improved_condition"
                    and "improved_condition" in filtered_data.columns
                ):
                    filtered_data = filtered_data[
                        filtered_data["improved_condition"].str.contains(
                            value, case=False, na=False
                        )
                    ]
                elif (
                    field == "min_condition_confidence"
                    and "condition_confidence" in filtered_data.columns
                ):
                    filtered_data = filtered_data[
                        filtered_data["condition_confidence"] >= value
                    ]

                # TDR and zoning criteria
                elif (
                    field == "tdr_eligible"
                    and "tdr_eligible_flag_lookup" in filtered_data.columns
                ):
                    filtered_data = filtered_data[
                        filtered_data["tdr_eligible_flag_lookup"] == value
                    ]
                elif (
                    field == "min_lot_buildout_gap"
                    and "lot_buildout_gap" in filtered_data.columns
                ):
                    filtered_data = filtered_data[
                        filtered_data["lot_buildout_gap"] >= value
                    ]
                elif field == "zoning_code" and "zoning_code" in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data["zoning_code"] == value]
                else:
                    # Direct field match if field exists in data
                    if field in filtered_data.columns:
                        filtered_data = filtered_data[filtered_data[field] == value]

            # Exclude the subject property if provided
            if subject_property is not None:
                # Try to exclude by address first
                if (
                    "full_address" in filtered_data.columns
                    and "full_address" in subject_property
                ):
                    subject_address = subject_property["full_address"]
                    filtered_data = filtered_data[
                        filtered_data["full_address"] != subject_address
                    ]

                # If no address or we still have the subject property, try by listing number
                if (
                    "list_number" in filtered_data.columns
                    and "list_number" in subject_property
                ):
                    subject_list_number = subject_property["list_number"]
                    filtered_data = filtered_data[
                        filtered_data["list_number"] != subject_list_number
                    ]

            # Sort the results
            if sort_by in filtered_data.columns:
                filtered_data = filtered_data.sort_values(
                    by=sort_by, ascending=ascending
                )

            # Check if we found enough comps
            if (
                len(filtered_data) >= min_comps
                or stage_idx == len(relaxation_stages) - 1
            ):
                logger.info(
                    f"Found {len(filtered_data)} comparable properties (stage {stage_idx + 1})"
                )

                if relaxed_filters:
                    logger.info(f"Filters relaxed: {', '.join(relaxed_filters)}")

                # Return limited results and relaxation info
                return filtered_data.head(limit), relaxed_filters

        # Should only get here if all stages failed
        return pd.DataFrame(), relaxed_filters

    def _relax_criteria(self, original_criteria, relaxations):
        """Helper method to relax criteria based on relaxation stage."""
        criteria = original_criteria.copy()

        if not relaxations:
            return criteria

        # Get the numeric relaxation factor
        relax_factor = relaxations.get("relax_numeric", 0)

        # Relax numeric ranges
        if relax_factor > 0:
            # Extend min/max ranges
            for (
                field
            ) in criteria.copy():  # Use copy to avoid modifying during iteration
                if field.startswith("min_"):
                    base_field = field[4:]  # Remove 'min_' prefix
                    if base_field in ["sqft", "bedrooms", "baths", "lot_buildout_gap"]:
                        criteria[field] = criteria[field] * (1 - relax_factor)
                elif field.startswith("max_"):
                    base_field = field[4:]  # Remove 'max_' prefix
                    if base_field in ["sqft", "bedrooms", "baths", "walk_to_gondola"]:
                        criteria[field] = criteria[field] * (1 + relax_factor)

        # Relax condition requirements
        if (
            relaxations.get("relax_condition", False)
            and "improved_condition" in criteria
        ):
            # Remove condition filter to allow any condition
            del criteria["improved_condition"]

        # Relax STR eligibility
        if relaxations.get("relax_str", False) and "str_eligible" in criteria:
            # Remove STR filter to allow any STR status
            del criteria["str_eligible"]

        # Relax property type
        if (
            relaxations.get("relax_property_type", False)
            and "property_type" in criteria
        ):
            # Remove property type filter to allow any property type
            del criteria["property_type"]

        return criteria

    def _get_relaxed_filter_names(self, relaxations):
        """Get human-readable names of relaxed filters."""
        relaxed = []

        if relaxations.get("relax_numeric", 0) > 0:
            pct = int(relaxations["relax_numeric"] * 100)
            relaxed.append(f"numeric_ranges_{pct}pct")

        if relaxations.get("relax_condition", False):
            relaxed.append("condition_requirements")

        if relaxations.get("relax_str", False):
            relaxed.append("str_eligibility")

        if relaxations.get("relax_property_type", False):
            relaxed.append("property_type")

        return relaxed

    def find_advanced_comps(
        self,
        subject_property,
        max_price_diff_pct=0.35,
        max_sqft_diff=500,
        max_demo_score_diff=1.0,
        min_comps=3,
        similarity_threshold=0.7,
        limit=10,
        additional_criteria=None,
    ):
        """
        Find comparable properties with advanced filtering logic to ensure true comparability.
        Uses value-based criteria to match similar properties beyond just basic attributes.

        Args:
            subject_property: Dict or Series with the subject property attributes
            max_price_diff_pct (float): Maximum price difference (as percentage) to allow
            max_sqft_diff (float): Maximum square footage difference to allow
            max_demo_score_diff (float): Maximum demo score difference to allow
            min_comps (int): Minimum number of comps to try to find
            similarity_threshold (float): Threshold for match score (0-1 scale)
            limit (int): Maximum number of comps to return
            additional_criteria (dict): Additional filtering criteria

        Returns:
            tuple: (DataFrame of comps, list of relaxed filters)
        """
        # Start with all data
        all_comps = self.data.copy()

        # Apply any additional basic criteria first (if provided)
        if additional_criteria:
            all_comps, _ = self.find_comps(
                criteria=additional_criteria,
                subject_property=subject_property,
                limit=1000,
            )

        # Extract subject property values (ensure it's a Series if passed as dict)
        if isinstance(subject_property, dict):
            subject_property = pd.Series(subject_property)

        # Get key attributes for the subject property
        s_price = (
            subject_property.get("asking_price")
            or subject_property.get("sold_price")
            or 0
        )
        s_price_per_sqft = subject_property.get("price_per_sqft", 0)
        s_sqft = subject_property.get("total_sqft", 0)
        s_bedrooms = subject_property.get("bedrooms", 0)
        s_baths = subject_property.get("total_baths", 0)
        s_condition = subject_property.get(
            "improved_condition",
            subject_property.get("property_condition", "Not Specified"),
        )
        s_demo_score = subject_property.get(
            "demo_score", 3.0
        )  # Default to middle of range if missing
        s_value_add = subject_property.get("value_add_type", "")
        s_str_eligible = subject_property.get("str_eligible", "")

        logger.info(
            f"Advanced filtering for property: {subject_property.get('full_address', 'Unknown')}"
        )
        logger.info(
            f"Base attributes: {s_bedrooms}BR/{s_baths}BA, {s_sqft} sqft, {s_condition} condition"
        )
        logger.info(
            f"Advanced attributes: Demo score {s_demo_score}, Value-add: {s_value_add}, STR: {s_str_eligible}"
        )

        # Define relaxation strategy
        original_price_diff = max_price_diff_pct
        original_sqft_diff = max_sqft_diff
        original_demo_diff = max_demo_score_diff

        # Track which filters were relaxed
        relaxed_filters = []

        # Try progressively relaxed filters until we get enough comps
        for attempt in range(3):  # Max 3 attempts with relaxed criteria
            # Basic filters to narrow down initial set
            filtered_comps = all_comps[
                (all_comps["bedrooms"] == s_bedrooms)
                & (all_comps["total_baths"] == s_baths)
                & (all_comps["total_sqft"] >= s_sqft - max_sqft_diff)
                & (all_comps["total_sqft"] <= s_sqft + max_sqft_diff)
            ].copy()

            # Exclude the subject property
            if "full_address" in filtered_comps.columns and hasattr(
                subject_property, "full_address"
            ):
                subject_address = subject_property["full_address"]
                filtered_comps = filtered_comps[
                    filtered_comps["full_address"] != subject_address
                ]

            # If no address or we still have the subject property, try by listing number
            if "list_number" in filtered_comps.columns and hasattr(
                subject_property, "list_number"
            ):
                subject_list_number = subject_property["list_number"]
                filtered_comps = filtered_comps[
                    filtered_comps["list_number"] != subject_list_number
                ]

            # Get count after basic filtering
            basic_filter_count = len(filtered_comps)

            # Apply advanced value-based filters
            if s_price > 0 and "price_per_sqft" in filtered_comps.columns:
                # Price range filter
                min_price = s_price * (1 - max_price_diff_pct)
                max_price = s_price * (1 + max_price_diff_pct)
                filtered_comps = filtered_comps[
                    (
                        (filtered_comps["asking_price"] >= min_price)
                        & (filtered_comps["asking_price"] <= max_price)
                    )
                    | (
                        (filtered_comps["sold_price"] >= min_price)
                        & (filtered_comps["sold_price"] <= max_price)
                    )
                ]

            # Get count after price filtering
            price_filter_count = len(filtered_comps)

            # Condition matching (if condition data is available)
            condition_groups = {
                "Excellent": [
                    "Excellent",
                    "Excellent Condition",
                    "Good",
                    "Good Condition",
                ],
                "Good": ["Good", "Good Condition", "Excellent", "Excellent Condition"],
                "Average": ["Average", "Average Condition", "Fair", "Fair Condition"],
                "Poor": ["Poor", "Poor Condition", "Fair", "Fair Condition"],
                "Under Construction": ["Under Construction", "New Build"],
                "New Build": ["New Build", "Under Construction"],
            }

            if (
                s_condition in condition_groups
                and "improved_condition" in filtered_comps.columns
                and attempt
                == 0  # Only apply strict condition matching on first attempt
            ):
                compatible_conditions = condition_groups.get(s_condition, [s_condition])
                condition_filter = filtered_comps["improved_condition"].isin(
                    compatible_conditions
                )

                # If we have few matches, also accept using property_condition
                if (
                    condition_filter.sum() < 5
                    and "property_condition" in filtered_comps.columns
                ):
                    condition_filter = condition_filter | filtered_comps[
                        "property_condition"
                    ].isin(compatible_conditions)

                filtered_comps = filtered_comps[condition_filter]

            # Get count after condition filtering
            condition_filter_count = len(filtered_comps)

            # Demo score matching (if available)
            if "demo_score" in filtered_comps.columns:
                filtered_comps = filtered_comps[
                    (filtered_comps["demo_score"] >= s_demo_score - max_demo_score_diff)
                    & (
                        filtered_comps["demo_score"]
                        <= s_demo_score + max_demo_score_diff
                    )
                ]

            # Get count after demo score filtering
            demo_filter_count = len(filtered_comps)

            # Value-add type matching (if available)
            if (
                s_value_add
                and "value_add_type" in filtered_comps.columns
                and attempt == 0
            ):
                # Group similar value-add types
                teardown_group = ["Teardown", "Gut Renovation"]
                cosmetic_group = ["Cosmetic Reno", "Hold"]

                if s_value_add in teardown_group:
                    filtered_comps = filtered_comps[
                        filtered_comps["value_add_type"].isin(teardown_group)
                    ]
                elif s_value_add in cosmetic_group:
                    filtered_comps = filtered_comps[
                        filtered_comps["value_add_type"].isin(cosmetic_group)
                    ]
                else:
                    filtered_comps = filtered_comps[
                        filtered_comps["value_add_type"] == s_value_add
                    ]

            # STR eligibility matching (if available)
            if (
                s_str_eligible
                and "str_eligible" in filtered_comps.columns
                and attempt == 0
            ):
                filtered_comps = filtered_comps[
                    filtered_comps["str_eligible"] == s_str_eligible
                ]

            # Get final count after all filters
            final_filter_count = len(filtered_comps)

            # Print filtering statistics
            logger.info("Filtering statistics:")
            logger.info(f"Initial comps after basic filtering: {basic_filter_count}")
            logger.info(f"After price range filter: {price_filter_count}")
            logger.info(f"After condition matching: {condition_filter_count}")
            logger.info(f"After demo score filtering: {demo_filter_count}")
            logger.info(f"Final comparable properties: {final_filter_count}")

            # Calculate match scores for remaining properties
            if not filtered_comps.empty:
                # Calculate a composite match score based on similarity to subject
                filtered_comps["size_match"] = 1 - abs(
                    filtered_comps["total_sqft"] - s_sqft
                ) / max(s_sqft, 1)

                if "demo_score" in filtered_comps.columns:
                    filtered_comps["demo_match"] = (
                        1 - abs(filtered_comps["demo_score"] - s_demo_score) / 5.0
                    )
                else:
                    filtered_comps["demo_match"] = 0.5  # Neutral score

                # Price similarity (if we have price data)
                if s_price_per_sqft > 0 and "price_per_sqft" in filtered_comps.columns:
                    filtered_comps["price_match"] = 1 - abs(
                        filtered_comps["price_per_sqft"] - s_price_per_sqft
                    ) / max(s_price_per_sqft, 1)
                else:
                    filtered_comps["price_match"] = 0.5  # Neutral score

                # Condition matching score
                filtered_comps["condition_match"] = filtered_comps.apply(
                    lambda row: 1.0
                    if row.get("improved_condition", "") == s_condition
                    else (
                        0.8
                        if row.get("improved_condition", "")
                        in condition_groups.get(s_condition, [])
                        else 0.5
                    ),
                    axis=1,
                )

                # Calculate composite match score (weighted)
                filtered_comps["match_score"] = (
                    filtered_comps["size_match"] * 0.25
                    + filtered_comps["demo_match"] * 0.25
                    + filtered_comps["price_match"] * 0.25
                    + filtered_comps["condition_match"] * 0.25
                )

                # Filter by match score threshold
                filtered_comps = filtered_comps[
                    filtered_comps["match_score"] >= similarity_threshold
                ]

                # Sort by match score (descending)
                filtered_comps = filtered_comps.sort_values(
                    "match_score", ascending=False
                )

            # If we have enough comps, we're done
            if len(filtered_comps) >= min_comps:
                break

            # If not enough comps and this isn't the last attempt, relax criteria
            if attempt < 2:  # Only relax if we have more attempts left
                if attempt == 0:
                    # First relaxation: widen price and sqft ranges
                    max_price_diff_pct *= 1.5  # 50% wider price range
                    max_sqft_diff *= 1.5  # 50% wider sqft range
                    relaxed_filters.append("price_range")
                    relaxed_filters.append("sqft_range")
                    logger.warning(
                        f"Relaxing price range to ±{max_price_diff_pct * 100:.0f}% and sqft range to ±{max_sqft_diff:.0f}"
                    )

                if attempt == 1:
                    # Second relaxation: relax demo score and ignore value-add matching
                    max_demo_score_diff *= 2.0  # Double demo score range
                    relaxed_filters.append("demo_score")
                    relaxed_filters.append("value_add_type")
                    logger.warning(
                        f"Relaxing demo score range to ±{max_demo_score_diff:.1f} and ignoring value-add type"
                    )

        # If we have too few comps after all attempts, log a warning
        if len(filtered_comps) < min_comps:
            logger.warning("Too few comparable properties found with strict matching.")
            logger.warning(
                "Consider relaxing criteria using find_comps() method instead."
            )

        # Return the top matches and the list of relaxed filters
        return filtered_comps.head(limit), relaxed_filters

    def run_comp_analysis_by_address(
        self,
        address_query,
        limit=5,
        min_comps=3,
        similarity_threshold=0.7,
        max_price_diff_pct=0.35,
        max_sqft_diff=500,
        export_results=True,
        export_dir=None,
    ):
        """
        Run a comprehensive property analysis based on an address search.

        Args:
            address_query (str): The address to search for
            limit (int): Maximum number of comps to return
            min_comps (int): Minimum number of comps to try to find
            similarity_threshold (float): Threshold for match score (0-1 scale)
            max_price_diff_pct (float): Maximum price difference (as percentage) to allow
            max_sqft_diff (float): Maximum square footage difference to allow
            export_results (bool): Whether to export results to CSV
            export_dir (str): Directory to export results to (default: current directory)

        Returns:
            dict: Analysis results including property details and comparable properties
        """
        # Find the property by address
        logger.info(f"Searching for property with address: {address_query}")
        subject_property = AddressMatcher.find_property_by_address(
            self.data, address_query
        )

        if subject_property is None or subject_property.empty:
            return {
                "error": f"No property found matching address: {address_query}",
                "success": False,
            }

        # Extract the property as a Series
        subject = subject_property.iloc[0]

        # Create summary of the subject property
        property_summary = {
            "address": subject.get("full_address", address_query),
            "bedrooms": subject.get("bedrooms", "N/A"),
            "bathrooms": subject.get("total_baths", "N/A"),
            "total_sqft": subject.get("total_sqft", "N/A"),
            "asking_price": subject.get(
                "asking_price", subject.get("sold_price", "N/A")
            ),
            "price_per_sqft": subject.get("price_per_sqft", "N/A"),
        }

        # Add enhanced data if available
        if self.has_demo_score:
            property_summary["demo_score"] = subject.get("demo_score", "N/A")
            property_summary["value_add_type"] = subject.get("value_add_type", "N/A")

        if self.has_condition_data:
            property_summary["condition"] = subject.get(
                "improved_condition", subject.get("property_condition", "N/A")
            )

        if self.has_tdr_data:
            property_summary["tdr_eligible"] = subject.get(
                "tdr_eligible_flag_lookup", "N/A"
            )
            property_summary["lot_buildout_gap"] = subject.get(
                "lot_buildout_gap", "N/A"
            )

        # Get STR eligibility if available
        if "str_eligible" in subject:
            property_summary["str_eligible"] = subject.get("str_eligible", "N/A")

        # Find comparable properties using advanced matching
        logger.info(f"Finding comparable properties for {property_summary['address']}")

        # First try advanced comps
        advanced_comps, relaxed_filters = self.find_advanced_comps(
            subject_property=subject,
            max_price_diff_pct=max_price_diff_pct,
            max_sqft_diff=max_sqft_diff,
            min_comps=min_comps,
            similarity_threshold=similarity_threshold,
            limit=limit,
        )

        # If we don't have enough advanced comps, try with basic criteria
        if len(advanced_comps) < min_comps:
            logger.info("Not enough advanced comps found, using basic criteria")
            basic_criteria = {
                "bedrooms": subject.get("bedrooms", 0),
                "min_sqft": max(0, subject.get("total_sqft", 0) - max_sqft_diff),
                "max_sqft": subject.get("total_sqft", 0) + max_sqft_diff,
            }

            if "improved_condition" in subject and pd.notna(
                subject["improved_condition"]
            ):
                basic_criteria["improved_condition"] = subject["improved_condition"]

            # Add area filter if available
            if "area" in subject and pd.notna(subject["area"]):
                basic_criteria["area"] = subject["area"]

            basic_comps, basic_relaxed = self.find_comps(
                criteria=basic_criteria,
                subject_property=subject,  # Exclude the subject property
                min_comps=min_comps,
                limit=limit,
                sort_by="match_score"
                if "match_score" in advanced_comps.columns
                else None,
                ascending=False,
            )
            relaxed_filters.append("basic_criteria_used")
            relaxed_filters.extend(basic_relaxed)
            comps = basic_comps
        else:
            comps = advanced_comps

        # Analyze the comparable properties
        comp_analysis = PropertyAnalyzer.analyze_comps(
            comps, time_adjust=self.time_adjust
        )

        # Compare to overall market
        market_comparison = PropertyAnalyzer.compare_to_market(
            comps, self.data, time_adjust=self.time_adjust
        )

        # Create result object
        result = {
            "success": True,
            "property": property_summary,
            "comparable_properties": comps.to_dict(orient="records")
            if not comps.empty
            else [],
            "analysis": comp_analysis,
            "market_comparison": market_comparison,
        }

        # Add filter relaxation info if any filters were relaxed
        if relaxed_filters:
            result["filter_relaxation_applied"] = relaxed_filters

        # Export results if requested
        if export_results and not comps.empty:
            # Create export directory if it doesn't exist
            export_path = export_dir if export_dir else "."
            try:
                import os

                if export_dir and not os.path.exists(export_dir):
                    os.makedirs(export_dir)

                safe_address = clean_filename(property_summary["address"])
                filename = f"{export_path}/comps_{safe_address}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
                export_file = PropertyAnalyzer.export_results(comps, filename)
                result["export_file"] = export_file
            except Exception as e:
                logger.error(f"Error exporting results: {str(e)}")
                # Continue even if export fails

        # Print summary to console
        PropertyAnalyzer.print_comp_analysis_summary(result)

        return result


def run_comp_analysis_by_address(
    address,
    csv_path=None,
    limit=5,
    min_comps=3,
    similarity_threshold=0.7,
    max_price_diff_pct=0.35,
    max_sqft_diff=500,
    export_results=True,
    export_dir=None,
):
    """
    Run a comprehensive property analysis based on an address search.
    This is a convenience function that creates an EnhancedCompFinder and runs the analysis.

    Args:
        address (str): The address to search for
        csv_path (str, optional): Path to the CSV file with property data
        limit (int): Maximum number of comps to return
        min_comps (int): Minimum number of comps to try to find
        similarity_threshold (float): Similarity threshold for matching (0-1)
        max_price_diff_pct (float): Maximum price difference percentage (0-1)
        max_sqft_diff (int): Maximum square footage difference to allow
        export_results (bool): Whether to export results to CSV
        export_dir (str): Directory to export results to

    Returns:
        dict: Analysis results including property details and comparable properties
    """
    # Create the comp finder
    finder = EnhancedCompFinder(csv_path=csv_path, time_adjust=True)

    # Run the analysis
    return finder.run_comp_analysis_by_address(
        address_query=address,
        limit=limit,
        min_comps=min_comps,
        similarity_threshold=similarity_threshold,
        max_price_diff_pct=max_price_diff_pct,
        max_sqft_diff=max_sqft_diff,
        export_results=export_results,
        export_dir=export_dir,
    )
