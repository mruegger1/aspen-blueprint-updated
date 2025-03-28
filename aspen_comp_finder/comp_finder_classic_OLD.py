#!/usr/bin/env python3
"""
Classic Comp Finder - Traditional Comparable Property Matcher
------------------------------------------------------------
Finds comparable properties based on standard buyer/appraiser criteria.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re

# Set up logging
logger = logging.getLogger("comp_finder_classic")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ClassicCompFinder:
    """Finds comparable properties using traditional matching criteria."""

    def __init__(self, csv_path=None, time_adjust=True):
        """
        Initialize the classic comp finder.

        Args:
            csv_path (str): Path to CSV file with property data
            time_adjust (bool): Whether to adjust prices for time
        """
        # Load data
        self.data, self.file_path = self._load_data(csv_path)
        self.time_adjust = time_adjust

        # Apply time-based adjustments if needed
        if time_adjust:
            self._adjust_prices_for_time()

    def _load_data(self, csv_path=None):
        """
        Load property data from CSV file.

        Args:
            csv_path (str): Path to CSV file

        Returns:
            tuple: (DataFrame of properties, path to file)
        """
        # Define potential CSV file locations
        potential_paths = []

        # Add the provided path if it exists
        if csv_path and os.path.exists(csv_path):
            potential_paths.append(csv_path)

        # Add default locations
        default_paths = [
            "aspen_mvp_final_scored.csv",
            "data/aspen_mvp_final_scored.csv",
            "../../Python Analysis/Data/mvp/aspen_mvp_final_scored.csv",
            "../data/aspen_mvp_final_scored.csv",
            os.path.expanduser(
                "~/Desktop/Aspen Real Estate Analysis Master/Python Analysis/Data/mvp/aspen_mvp_final_scored.csv"
            ),
        ]

        for path in default_paths:
            if os.path.exists(path):
                potential_paths.append(path)

        # Try to load from the first available path
        if not potential_paths:
            raise FileNotFoundError(
                "Could not find property data CSV file. Please provide a valid path."
            )

        # Load the first available file
        file_path = potential_paths[0]
        logger.info(f"Loading data from: {file_path}")

        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} properties from CSV")

            # Ensure critical columns exist
            required_cols = [
                "bedrooms",
                "total_baths",
                "adjusted_sold_price",
                "total_sqft",
                "sold_date",
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                logger.warning(f"Missing required columns: {', '.join(missing_cols)}")

            # Convert date columns
            date_columns = ["sold_date", "list_date", "close_date"]
            for col in date_columns:
                if col in data.columns:
                    data[f"{col}_dt"] = pd.to_datetime(data[col], errors="coerce")

            return data, file_path

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _adjust_prices_for_time(self):
        """Apply time-based price adjustments to account for market appreciation/depreciation."""
        logger.info("=== Price Adjustment For Time ===")

        # Define annual appreciation rates (customize these based on market data)
        appreciation_rates = {
            2021: 0.18,  # 18% appreciation
            2022: 0.12,  # 12% appreciation
            2023: 0.09,  # 9% appreciation
            2024: 0.06,  # 6% appreciation
            2025: 0.00,  # Current year (reference point)
        }

        # Log the rates
        logger.info(
            "Applying the following appreciation rates to normalize historical prices:"
        )
        for year, rate in appreciation_rates.items():
            logger.info(f"{year}: {rate:.1%}")

        # Create adjusted price columns if they don't exist
        if "adjusted_sold_price_time" not in self.data.columns:
            self.data["adjusted_sold_price_time"] = self.data[
                "adjusted_sold_price"
            ].copy()

        if "adjusted_price_per_sqft_time" not in self.data.columns:
            # Make sure we have price_per_sqft column
            if (
                "price_per_sqft" not in self.data.columns
                and "total_sqft" in self.data.columns
            ):
                self.data["price_per_sqft"] = (
                    self.data["adjusted_sold_price"] / self.data["total_sqft"]
                )

            self.data["adjusted_price_per_sqft_time"] = self.data[
                "price_per_sqft"
            ].copy()

        # Apply time adjustments based on sold_date
        current_year = datetime.now().year

        # Process each row
        for idx, row in self.data.iterrows():
            if pd.notna(row.get("sold_date_dt")):
                sold_year = row["sold_date_dt"].year

                # Calculate cumulative adjustment factor
                adjustment_factor = 1.0
                for year in range(sold_year, current_year):
                    if year in appreciation_rates:
                        adjustment_factor *= 1 + appreciation_rates[year]

                # Apply adjustment
                self.data.at[idx, "adjusted_sold_price_time"] = (
                    row["adjusted_sold_price"] * adjustment_factor
                )
                if "price_per_sqft" in self.data.columns and pd.notna(
                    row["price_per_sqft"]
                ):
                    self.data.at[idx, "adjusted_price_per_sqft_time"] = (
                        row["price_per_sqft"] * adjustment_factor
                    )

    def find_classic_comps(
        self,
        bedrooms=None,
        bathrooms=None,
        property_type=None,
        area=None,
        str_eligible=None,
        condition=None,
        max_price=None,
        sqft_min=None,
        sqft_max=None,
        year_built_min=None,
        months_back=24,
        listing_status=None,  # Can be "A" (Active), "P" (Pending), "S" (Sold)
        limit=5,
        export_results=False,
        export_dir="outputs",
        export_filename=None,
    ):
        """
        Find comparable properties based on classic buyer/appraiser criteria.

        Args:
            bedrooms (int): Number of bedrooms to match
            bathrooms (float): Number of bathrooms to match (can be .5 increments)
            property_type (str): Type of property (condo, townhome, single-family)
            area (str): Neighborhood or area
            str_eligible (bool): Whether short-term rental is allowed
            condition (str): Property condition (Excellent, Good, Average)
            max_price (float): Maximum price to consider
            sqft_min (float): Minimum square footage
            sqft_max (float): Maximum square footage
            year_built_min (int): Minimum year built
            months_back (int): How many months of sales to consider
            limit (int): Maximum number of comps to return
            export_results (bool): Whether to export results to CSV
            export_dir (str): Directory for export files
            export_filename (str): Custom filename for export

        Returns:
            dict: Results containing matched comps and statistics
        """
        # Build criteria description for logging
        criteria_parts = []
        if bedrooms is not None:
            criteria_parts.append(f"{bedrooms}BR")
        if bathrooms is not None:
            criteria_parts.append(f"{bathrooms}BA")
        if area is not None:
            criteria_parts.append(f"in {area}")
        if listing_status is not None:
            status_desc = {"A": "Active", "P": "Pending", "S": "Sold"}.get(
                listing_status, listing_status
            )
            criteria_parts.append(f"({status_desc})")

        criteria_str = "/".join(criteria_parts)
        logger.info(f"Finding classic comps with criteria: {criteria_str}")

        # Make a copy of the data to work with
        df = self.data.copy()

        # Filter by listing status if specified
        if listing_status is not None and "listing_status" in df.columns:
            df = df[df["listing_status"] == listing_status]
            logger.info(
                f"Filtered to {len(df)} properties with listing status: {listing_status}"
            )

        # Filter for recent sales (only if we're looking at sold properties)
        if (
            listing_status in [None, "S", "Sold", "SOLD"]
            and "sold_date_dt" in df.columns
        ):
            cutoff_date = datetime.now() - timedelta(days=30 * months_back)
            df = df[df["sold_date_dt"] >= cutoff_date]
            logger.info(
                f"Filtered to {len(df)} properties sold in the last {months_back} months"
            )

        # Apply basic filters first (these must match exactly)
        initial_count = len(df)

        # Basic filters
        if bedrooms is not None:
            df = df[df["bedrooms"] == bedrooms]

        if bathrooms is not None:
            df = df[df["total_baths"] == bathrooms]

        if max_price is not None:
            price_col = (
                "adjusted_sold_price_time"
                if "adjusted_sold_price_time" in df.columns
                else "adjusted_sold_price"
            )
            df = df[df[price_col] <= max_price]

        if year_built_min is not None and "year_built" in df.columns:
            df = df[df["year_built"] >= year_built_min]

        logger.info(f"After basic filters: {len(df)} properties")

        if len(df) == 0:
            logger.warning(
                "No properties match the basic criteria. Try relaxing filters."
            )
            return {
                "comps": pd.DataFrame(),
                "stats": {},
                "filter_stats": {
                    "initial_count": initial_count,
                    "after_basic_filters": 0,
                },
                "message": "No properties match the basic criteria. Try relaxing filters.",
            }

        # Create a scoring system for remaining properties
        df["match_score"] = 0.0

        # Define weights for different criteria
        weights = {
            "bedrooms": 2.0,
            "bathrooms": 2.0,
            "property_type": 2.0,
            "area": 1.5,
            "condition": 1.5,
            "sqft": 1.0,
            "str": 1.0,
            "recency": 1.0,
        }

        # Score bedrooms match
        if bedrooms is not None:
            df["bedroom_score"] = 1.0 - (0.25 * np.abs(df["bedrooms"] - bedrooms))
            df["bedroom_score"] = df["bedroom_score"].clip(0, 1)
            df["match_score"] += weights["bedrooms"] * df["bedroom_score"]

        # Score bathrooms match
        if bathrooms is not None:
            df["bathroom_score"] = 1.0 - (0.25 * np.abs(df["total_baths"] - bathrooms))
            df["bathroom_score"] = df["bathroom_score"].clip(0, 1)
            df["match_score"] += weights["bathrooms"] * df["bathroom_score"]

        # Score property type match
        if property_type is not None and "resolved_property_type" in df.columns:
            # Exact match gets 1.0, similar types get partial scores
            property_type_map = {
                "Condo": ["Condo", "Condominium", "Apartment"],
                "Townhouse": ["Townhouse", "Townhome", "Town House", "Townhome/Condo"],
                "Single Family": [
                    "Single Family",
                    "House",
                    "Detached",
                    "Single-Family",
                ],
            }

            # Normalize input property type
            normalized_type = None
            for category, variants in property_type_map.items():
                if property_type.lower() in [v.lower() for v in variants]:
                    normalized_type = category
                    break

            if normalized_type:
                df["property_type_score"] = 0.0
                for category, variants in property_type_map.items():
                    match_score = 1.0 if category == normalized_type else 0.3
                    mask = (
                        df["resolved_property_type"]
                        .str.lower()
                        .isin([v.lower() for v in variants])
                    )
                    df.loc[mask, "property_type_score"] = match_score
            else:
                # If property_type doesn't match any category, do direct comparison
                df["property_type_score"] = np.where(
                    df["resolved_property_type"].str.lower() == property_type.lower(),
                    1.0,
                    0.0,
                )

            df["match_score"] += weights["property_type"] * df["property_type_score"]

        # Score area match
        if area is not None and "area" in df.columns:
            # Define area proximity map (areas that are close to each other)
            area_proximity = {
                "Core": ["Core", "Downtown", "Central Core"],
                "West End": ["West End", "West Side"],
                "East End": ["East End", "East Side"],
                "Red Mountain": ["Red Mountain", "Red Mtn"],
                "Smuggler": ["Smuggler", "Smuggler Mountain"],
                "McLain Flats": ["McLain Flats", "McLain"],
                "Woody Creek": ["Woody Creek"],
                "Starwood": ["Starwood"],
            }

            # Normalize input area
            normalized_area = None
            for category, variants in area_proximity.items():
                if area.lower() in [v.lower() for v in variants]:
                    normalized_area = category
                    break

            if normalized_area:
                df["area_score"] = 0.0
                for category, variants in area_proximity.items():
                    # Exact area match gets 1.0, nearby areas get 0.7
                    match_score = 1.0 if category == normalized_area else 0.0

                    # Define adjacent areas (customize based on Aspen geography)
                    adjacent_areas = {
                        "Core": ["West End", "East End"],
                        "West End": ["Core", "Red Mountain"],
                        "East End": ["Core", "Smuggler"],
                        "Red Mountain": ["West End"],
                        "Smuggler": ["East End"],
                    }

                    # If areas are adjacent, give partial score
                    if (
                        category != normalized_area
                        and normalized_area in adjacent_areas
                        and category in adjacent_areas[normalized_area]
                    ):
                        match_score = 0.7

                    mask = df["area"].str.lower().isin([v.lower() for v in variants])
                    df.loc[mask, "area_score"] = match_score
            else:
                # Direct comparison if no match in map
                df["area_score"] = np.where(
                    df["area"].str.lower() == area.lower(), 1.0, 0.0
                )

            df["match_score"] += weights["area"] * df["area_score"]

        # Score condition match
        if condition is not None and "improved_condition" in df.columns:
            condition_values = ["Excellent", "Good", "Average", "Fair", "Poor"]

            # Find the index of the target condition
            if condition in condition_values:
                target_idx = condition_values.index(condition)

                # Score based on distance in the condition scale
                df["condition_score"] = 0.0
                for idx, row in df.iterrows():
                    if (
                        pd.isna(row["improved_condition"])
                        or row["improved_condition"] not in condition_values
                    ):
                        df.at[idx, "condition_score"] = (
                            0.5  # Default score for unknown condition
                        )
                    else:
                        prop_idx = condition_values.index(row["improved_condition"])
                        distance = abs(target_idx - prop_idx)

                        # Score decreases with distance from target condition
                        if distance == 0:
                            df.at[idx, "condition_score"] = 1.0
                        elif distance == 1:
                            df.at[idx, "condition_score"] = 0.8
                        else:
                            df.at[idx, "condition_score"] = max(
                                0, 1.0 - (distance * 0.3)
                            )
            else:
                # Direct comparison if condition not in scale
                df["condition_score"] = np.where(
                    df["improved_condition"] == condition, 1.0, 0.5
                )

            df["match_score"] += weights["condition"] * df["condition_score"]

        # Score square footage match
        if "total_sqft" in df.columns and (
            sqft_min is not None or sqft_max is not None
        ):
            # Determine target square footage
            if sqft_min is not None and sqft_max is not None:
                target_sqft = (sqft_min + sqft_max) / 2
            elif sqft_min is not None:
                target_sqft = sqft_min * 1.1  # Assume 10% more than minimum
            elif sqft_max is not None:
                target_sqft = sqft_max * 0.9  # Assume 10% less than maximum

            # Score based on percentage difference
            df["sqft_score"] = 0.0
            for idx, row in df.iterrows():
                if pd.isna(row["total_sqft"]) or row["total_sqft"] == 0:
                    df.at[idx, "sqft_score"] = 0.5  # Default score for missing data
                else:
                    pct_diff = abs(row["total_sqft"] - target_sqft) / target_sqft

                    # Higher score for closer size match
                    if pct_diff <= 0.05:  # Within 5%
                        df.at[idx, "sqft_score"] = 1.0
                    elif pct_diff <= 0.10:  # Within 10%
                        df.at[idx, "sqft_score"] = 0.9
                    elif pct_diff <= 0.20:  # Within 20%
                        df.at[idx, "sqft_score"] = 0.8
                    elif pct_diff <= 0.30:  # Within 30%
                        df.at[idx, "sqft_score"] = 0.6
                    else:
                        df.at[idx, "sqft_score"] = max(0, 1.0 - pct_diff)

            df["match_score"] += weights["sqft"] * df["sqft_score"]

        # Score STR eligibility match
        if str_eligible is not None and "str_eligible" in df.columns:
            # Convert to standardized format
            if isinstance(str_eligible, str):
                str_eligible = str_eligible.lower() in ["yes", "true", "y", "1"]

            df["str_score"] = np.where(
                df["str_eligible"]
                .astype(str)
                .str.lower()
                .isin(["yes", "true", "y", "1"])
                == str_eligible,
                1.0,
                0.0,
            )

            df["match_score"] += weights["str"] * df["str_score"]

        # Score recency
        if "sold_date_dt" in df.columns:
            # More recent sales get higher scores
            now = datetime.now()
            df["days_since_sale"] = (now - df["sold_date_dt"]).dt.days

            # Normalize to 0-1 range (1 = most recent, 0 = oldest in range)
            max_days = df["days_since_sale"].max()
            min_days = df["days_since_sale"].min()

            if max_days > min_days:
                df["recency_score"] = 1.0 - (
                    (df["days_since_sale"] - min_days) / (max_days - min_days)
                )
            else:
                df["recency_score"] = 1.0

            df["match_score"] += weights["recency"] * df["recency_score"]

        # Normalize match score to 0-100 range for readability
        total_weight = sum(weights.values())
        df["match_score"] = (df["match_score"] / total_weight) * 100

        # Sort by match score and take top results
        comps = df.sort_values("match_score", ascending=False).head(limit)

        logger.info(f"Found {len(comps)} comparable properties")

        # Prepare statistics
        if len(comps) > 0 and "adjusted_sold_price_time" in comps.columns:
            avg_price = comps["adjusted_sold_price_time"].mean()
            median_price = comps["adjusted_sold_price_time"].median()
            price_range = (
                comps["adjusted_sold_price_time"].min(),
                comps["adjusted_sold_price_time"].max(),
            )

            stats = {
                "average_price": avg_price,
                "median_price": median_price,
                "price_range": price_range,
            }

            if "adjusted_price_per_sqft_time" in comps.columns:
                avg_price_sqft = comps["adjusted_price_per_sqft_time"].mean()
                median_price_sqft = comps["adjusted_price_per_sqft_time"].median()
                price_sqft_range = (
                    comps["adjusted_price_per_sqft_time"].min(),
                    comps["adjusted_price_per_sqft_time"].max(),
                )

                stats.update(
                    {
                        "average_price_per_sqft": avg_price_sqft,
                        "median_price_per_sqft": median_price_sqft,
                        "price_per_sqft_range": price_sqft_range,
                    }
                )
        else:
            stats = {}

        # Export results if requested
        if export_results and len(comps) > 0:
            self._export_results(
                comps, export_dir, export_filename, bedrooms, bathrooms, area
            )

        # Return results
        return {
            "comps": comps,
            "stats": stats,
            "filter_stats": {
                "initial_count": initial_count,
                "after_basic_filters": len(df),
                "final_comps": len(comps),
            },
        }

    def _export_results(
        self,
        comps,
        export_dir,
        export_filename=None,
        bedrooms=None,
        bathrooms=None,
        area=None,
    ):
        """Export results to CSV file."""
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)

        # Generate filename if not provided
        if not export_filename:
            timestamp = datetime.now().strftime("%Y%m%d")
            criteria = []

            if bedrooms is not None:
                criteria.append(f"{bedrooms}BR")

            if bathrooms is not None:
                criteria.append(f"{bathrooms}BA")

            if area:
                # Remove spaces and special characters
                clean_area = re.sub(r"[^a-zA-Z0-9]", "_", area)
                criteria.append(clean_area)

            criteria_str = "_".join(criteria) if criteria else "all"
            export_filename = f"classic_comps_{criteria_str}_{timestamp}.csv"

        # Ensure file has .csv extension
        if not export_filename.endswith(".csv"):
            export_filename += ".csv"

        # Full path for export
        export_path = os.path.join(export_dir, export_filename)

        # Export the DataFrame
        comps.to_csv(export_path, index=False)
        logger.info(f"Results exported to {export_path}")

        return export_path


def run_classic_comp_search(
    bedrooms=None,
    bathrooms=None,
    property_type=None,
    area=None,
    str_eligible=None,
    condition=None,
    max_price=None,
    sqft_min=None,
    sqft_max=None,
    year_built_min=None,
    months_back=24,
    listing_status=None,  # Can be "A" (Active), "P" (Pending), "S" (Sold)
    reference_property_address=None,  # Can search for comps similar to a specific property
    limit=5,
    csv_path=None,
    export_results=True,
    export_dir="outputs",
):
    """
    Run a classic comparable property search.

    Args:
        (same as find_classic_comps method)

    Returns:
        dict: Results dictionary
    """
    try:
        # Initialize the finder
        finder = ClassicCompFinder(csv_path=csv_path, time_adjust=True)

        # If a reference property is provided, find its details to use as search criteria
        if reference_property_address:
            logger.info(
                f"Finding comps relative to reference property: {reference_property_address}"
            )
            # Find the reference property in the dataset
            reference_props = finder.data[
                finder.data["full_address"].str.contains(
                    reference_property_address, case=False
                )
            ]

            if len(reference_props) == 0:
                # Try alternative address fields
                for addr_field in ["street_address", "address", "location"]:
                    if addr_field in finder.data.columns:
                        reference_props = finder.data[
                            finder.data[addr_field].str.contains(
                                reference_property_address, case=False
                            )
                        ]
                        if len(reference_props) > 0:
                            break

            if len(reference_props) > 0:
                reference_prop = reference_props.iloc[0]
                logger.info(
                    f"Found reference property: {reference_prop.get('full_address', reference_property_address)}"
                )

                # Use the reference property's attributes if not explicitly provided
                if bedrooms is None and "bedrooms" in reference_prop:
                    bedrooms = reference_prop["bedrooms"]
                    logger.info(f"Using reference property bedrooms: {bedrooms}")

                if bathrooms is None and "total_baths" in reference_prop:
                    bathrooms = reference_prop["total_baths"]
                    logger.info(f"Using reference property bathrooms: {bathrooms}")

                if property_type is None and "resolved_property_type" in reference_prop:
                    property_type = reference_prop["resolved_property_type"]
                    logger.info(f"Using reference property type: {property_type}")

                if area is None and "area" in reference_prop:
                    area = reference_prop["area"]
                    logger.info(f"Using reference property area: {area}")

                if condition is None and "improved_condition" in reference_prop:
                    condition = reference_prop["improved_condition"]
                    logger.info(f"Using reference property condition: {condition}")

                if str_eligible is None and "str_eligible" in reference_prop:
                    str_eligible = reference_prop["str_eligible"]
                    logger.info(
                        f"Using reference property STR eligibility: {str_eligible}"
                    )

                if (
                    sqft_min is None
                    and sqft_max is None
                    and "total_sqft" in reference_prop
                ):
                    # Use ±20% of the reference property's square footage
                    ref_sqft = reference_prop["total_sqft"]
                    sqft_min = ref_sqft * 0.8
                    sqft_max = ref_sqft * 1.2
                    logger.info(
                        f"Using reference property sqft range: {sqft_min:.0f} - {sqft_max:.0f}"
                    )
            else:
                logger.warning(
                    f"Reference property not found: {reference_property_address}"
                )

        # Run the search
        results = finder.find_classic_comps(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            property_type=property_type,
            area=area,
            str_eligible=str_eligible,
            condition=condition,
            max_price=max_price,
            sqft_min=sqft_min,
            sqft_max=sqft_max,
            year_built_min=year_built_min,
            months_back=months_back,
            listing_status=listing_status,
            limit=limit,
            export_results=export_results,
            export_dir=export_dir,
        )

        # Print summary to console
        comps = results["comps"]
        stats = results["stats"]

        print("\n" + "=" * 80)
        criteria_list = []
        if bedrooms is not None:
            criteria_list.append(f"{bedrooms}BR")
        if bathrooms is not None:
            criteria_list.append(f"{bathrooms}BA")
        if property_type:
            criteria_list.append(property_type)
        if area:
            criteria_list.append(f"in {area}")
        if str_eligible is not None:
            criteria_list.append("STR-eligible" if str_eligible else "non-STR")

        # Add listing status to criteria
        status_label = ""
        if listing_status == "A":
            status_label = "ACTIVE LISTINGS"
        elif listing_status == "P":
            status_label = "PENDING LISTINGS"
        elif listing_status in ["S", "Sold", "SOLD"]:
            status_label = "SOLD PROPERTIES"

        criteria_str = " ".join(criteria_list)

        if reference_property_address:
            print(
                f"CLASSIC COMP FINDER: Properties similar to {reference_property_address}"
            )
            if status_label:
                print(f"{status_label} - {criteria_str}")
        else:
            if status_label:
                print(f"CLASSIC COMP FINDER: {status_label}")
                print(f"Criteria: {criteria_str}")
            else:
                print(f"CLASSIC COMP FINDER: {criteria_str}")

        print("=" * 80)

        if len(comps) == 0:
            print("\nNo comparable properties found matching your criteria.")
            print("Try relaxing some filters or expanding your search area.")
            return results

        print(
            f"\nFound {len(comps)} comparable properties from the past {months_back} months"
        )

        if stats:
            # Determine which price field to use (list price for active, sold price for sold)
            price_field = "average_price"
            price_sqft_field = "average_price_per_sqft"
            price_label = "Price"

            # Adjust labels based on listing status
            if listing_status == "A":
                price_label = "List Price"
            elif listing_status in ["S", "Sold", "SOLD"]:
                price_label = "Sold Price"

            print(f"\n{price_label.upper()} STATISTICS:")
            if price_field in stats:
                print(f"  Average {price_label}: ${stats[price_field]:,.0f}")
            if "median_price" in stats:
                print(f"  Median {price_label}: ${stats['median_price']:,.0f}")
            if "price_range" in stats:
                print(
                    f"  {price_label} Range: ${stats['price_range'][0]:,.0f} - ${stats['price_range'][1]:,.0f}"
                )

            if price_sqft_field in stats:
                print(f"\n{price_label.upper()} PER SQFT STATISTICS:")
                print(f"  Average: ${stats[price_sqft_field]:,.2f}/sqft")
                print(f"  Median: ${stats['median_price_per_sqft']:,.2f}/sqft")
                print(
                    f"  Range: ${stats['price_per_sqft_range'][0]:,.2f} - ${stats['price_per_sqft_range'][1]:,.2f}/sqft"
                )

        # Display top comps
        print("\nTOP COMPARABLE PROPERTIES:")

        # Determine display columns
        display_cols = []

        # Core columns
        display_cols.extend(
            ["full_address", "sold_date", "adjusted_sold_price_time", "match_score"]
        )

        # Add filtered criteria columns
        if bedrooms is not None:
            display_cols.append("bedrooms")
        if bathrooms is not None:
            display_cols.append("total_baths")
        if property_type is not None:
            display_cols.append("resolved_property_type")
        if area is not None:
            display_cols.append("area")
        if condition is not None:
            display_cols.append("improved_condition")
        if str_eligible is not None:
            display_cols.append("str_eligible")
        if sqft_min is not None or sqft_max is not None:
            display_cols.append("total_sqft")

        # Make sure we have all columns before displaying
        display_cols = [col for col in display_cols if col in comps.columns]

        # Display top properties
        for idx, comp in comps.iterrows():
            print(f"\n{idx + 1}. {comp.get('full_address', 'Address not available')}")
            print(f"   Match Score: {comp.get('match_score', 0):.1f}")

            # Display price with time adjustment note
            if "adjusted_sold_price_time" in comp:
                original_price = comp.get(
                    "adjusted_sold_price", comp["adjusted_sold_price_time"]
                )
                print(f"   Price: ${comp['adjusted_sold_price_time']:,.0f}")

                if abs(original_price - comp["adjusted_sold_price_time"]) > 100:
                    print(f"   (Original: ${original_price:,.0f}, adjusted for time)")

            # Display key attributes
            attrs = []
            if "bedrooms" in comp:
                attrs.append(f"{int(comp['bedrooms'])}BR")
            if "total_baths" in comp:
                attrs.append(f"{comp['total_baths']}BA")
            if "total_sqft" in comp:
                attrs.append(f"{int(comp['total_sqft'])} sqft")
            if attrs:
                print(f"   {' | '.join(attrs)}")

            # Display property type and condition
            type_cond = []
            if "resolved_property_type" in comp:
                type_cond.append(comp["resolved_property_type"])
            if "improved_condition" in comp:
                type_cond.append(comp["improved_condition"])
            if type_cond:
                print(f"   {' | '.join(type_cond)}")

            # Show STR eligibility
            if "str_eligible" in comp:
                print(f"   STR Eligible: {comp['str_eligible']}")

            # Show sale date
            if "sold_date" in comp:
                print(f"   Sold: {comp['sold_date']}")

        # Note export location
        if export_results:
            if "export_path" in results:
                print(f"\nResults exported to: {results['export_path']}")
            else:
                print(f"\nResults exported to: {export_dir}")

        return results

    except Exception as e:
        logger.error(f"Error in classic comp search: {str(e)}")
        import traceback

        traceback.print_exc()

        return {
            "error": str(e),
            "comps": pd.DataFrame(),
            "stats": {},
            "filter_stats": {},
        }


if __name__ == "__main__":
    # Simple test of the functionality
    run_classic_comp_search(
        bedrooms=2,
        bathrooms=2,
        property_type="Condo",
        area="Core",
        str_eligible=True,
        months_back=36,
        limit=5,
    )
