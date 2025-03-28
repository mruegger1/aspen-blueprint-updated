Here’s the full code: 
#!/usr/bin/env python3
"""
Property features extraction and calculation for real estate data
"""

import pandas as pd
import numpy as np
import os
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def categorize_property_condition(features):
    """
    Extract property condition from features string

    Args:
        features (str): Features string

    Returns:
        str: Property condition category
    """
    features = str(features)

    if "Under Construction" in features:
        return "Under Construction"
    elif "Condition|Excellent" in features:
        return "Excellent"
    elif "Condition|Good" in features:
        return "Good"
    elif "Condition|Average" in features:
        return "Average"
    elif "Condition|New" in features:
        return "New Build"
    else:
        return "Not Specified"


def extract_property_condition(df):
    """
    Extract property condition from features column

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with property_condition column
    """
    if "features" not in df.columns:
        logger.warning("No 'features' column found, skipping condition extraction")
        df["property_condition"] = "Unknown"
        return df

    logger.info("Extracting property condition from features")
    df["property_condition"] = df["features"].apply(categorize_property_condition)

    # Check alignment with new_construction flag
    if "new_construction" in df.columns:
        # If new_construction is Yes but property_condition isn't New Build, log a warning
        misaligned = (
            (df["new_construction"] == "Yes")
            & (df["property_condition"] != "New Build")
        ).sum()
        logger.info(
            f"Found {misaligned} properties marked as new construction but not categorized as New Build"
        )

    logger.info(
        f"Property condition distribution: {df['property_condition'].value_counts().to_dict()}"
    )
    return df


def calculate_bath_score(df):
    """
    Calculate bath score from component bath counts

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with bath_score column
    """
    # Check for required columns
    bath_columns = ["full_baths", "half_baths", "three_quarter_baths"]
    missing_columns = [col for col in bath_columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Missing bath columns: {missing_columns}")

        # Try to find alternative column names
        column_alts = {
            "full_baths": ["full_bath", "baths_full", "baths_1"],
            "half_baths": ["half_bath", "baths_half", "baths_0.5"],
            "three_quarter_baths": ["three_quarter_bath", "baths_3_4", "baths_0.75"],
        }

        for missing in missing_columns:
            for alt in column_alts[missing]:
                if alt in df.columns:
                    logger.info(f"Using alternative column {alt} for {missing}")
                    df[missing] = df[alt]
                    break

    # Fill missing values with 0
    for col in bath_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Calculate bath score
    logger.info("Calculating bath score")
    df["bath_score"] = (
        df["full_baths"] + (df["half_baths"] * 0.5) + (df["three_quarter_baths"] * 0.75)
    )

    # If total_baths exists, verify our calculation
    if "total_baths" in df.columns:
        # Calculate absolute difference between bath_score and total_baths
        df["bath_diff"] = (df["bath_score"] - df["total_baths"]).abs()
        mean_diff = df["bath_diff"].mean()
        logger.info(
            f"Mean difference between calculated bath_score and total_baths: {mean_diff:.4f}"
        )

        # If difference is significant, use total_baths where they differ by more than 0.5
        if mean_diff > 0.1:
            significant_diff = (df["bath_diff"] > 0.5).sum()
            logger.warning(
                f"Found {significant_diff} properties with significant bath count differences"
            )

            # Only replace where difference is significant
            mask = df["bath_diff"] > 0.5
            df.loc[mask, "bath_score"] = df.loc[mask, "total_baths"]

        # Remove temporary column
        df = df.drop(columns=["bath_diff"])

    return df


def calculate_below_ground_metrics(df):
    """
    Calculate below ground square footage metrics

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with below ground metrics
    """
    # Check for required columns
    if "below_ground_sqft" not in df.columns:
        logger.warning("No below_ground_sqft column found")

        # Look for alternative column
        alt_cols = ["LvHtSqFt (Blw Grnd)", "sqft_below_ground", "basement_sqft"]
        for col in alt_cols:
            if col in df.columns:
                logger.info(f"Using {col} for below_ground_sqft")
                df["below_ground_sqft"] = df[col]
                break
        else:
            logger.warning(
                "Could not find below ground square footage column, skipping below ground metrics"
            )
            return df

    if "total_sqft" not in df.columns:
        logger.warning(
            "No total_sqft column found, cannot calculate below ground ratio"
        )
        return df

    # Calculate below ground ratio
    logger.info("Calculating below ground metrics")

    # Ensure columns are numeric
    df["below_ground_sqft"] = pd.to_numeric(
        df["below_ground_sqft"], errors="coerce"
    ).fillna(0)
    df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")

    # Calculate ratio
    mask = (df["total_sqft"] > 0) & (df["below_ground_sqft"] > 0)
    df.loc[mask, "below_ground_ratio"] = (
        df.loc[mask, "below_ground_sqft"] / df.loc[mask, "total_sqft"]
    )

    # Flag properties with significant below ground space (>25%)
    df["has_significant_below_ground"] = (df["below_ground_ratio"] > 0.25).astype(str)
    df.loc[
        df["has_significant_below_ground"] == "True", "has_significant_below_ground"
    ] = "Yes"
    df.loc[
        df["has_significant_below_ground"] == "False", "has_significant_below_ground"
    ] = "No"

    # Count properties with significant below ground space
    significant_count = (df["has_significant_below_ground"] == "Yes").sum()
    logger.info(
        f"Found {significant_count} properties with significant below ground space"
    )

    return df


def calculate_lot_size_metrics(df):
    """
    Calculate lot size metrics and categories

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with lot size metrics
    """
    # Check for lot square footage column
    if "lot_sqft" not in df.columns:
        logger.warning("No lot_sqft column found")

        # Look for alternative column
        alt_cols = ["Lot SqFt", "lot_square_feet"]
        for col in alt_cols:
            if col in df.columns:
                logger.info(f"Using {col} for lot_sqft")
                df["lot_sqft"] = df[col]
                break

    # Check for acres column
    if "acres" not in df.columns:
        logger.warning("No acres column found")

        # Look for alternative column
        alt_cols = ["Nbr of Acres", "acreage"]
        for col in alt_cols:
            if col in df.columns:
                logger.info(f"Using {col} for acres")
                df["acres"] = df[col]
                break

    # Convert lot_sqft to numeric if it exists
    if "lot_sqft" in df.columns:
        df["lot_sqft"] = pd.to_numeric(df["lot_sqft"], errors="coerce")

    # Convert acres to numeric if it exists
    if "acres" in df.columns:
        df["acres"] = pd.to_numeric(df["acres"], errors="coerce")

    # Cross-check lot_sqft with acres
    if "lot_sqft" in df.columns and "acres" in df.columns:
        logger.info("Cross-checking lot_sqft with acres")

        # Calculate lot_sqft from acres where lot_sqft is missing
        mask = (df["lot_sqft"].isna() | (df["lot_sqft"] == 0)) & (df["acres"] > 0)
        df.loc[mask, "lot_sqft"] = df.loc[mask, "acres"] * 43560  # 1 acre = 43,560 sqft

        # Flag records where both are present but significantly different
        mask = (df["lot_sqft"] > 0) & (df["acres"] > 0)
        df.loc[mask, "calculated_lot_sqft"] = df.loc[mask, "acres"] * 43560
        df.loc[mask, "lot_size_diff_pct"] = (
            (df.loc[mask, "lot_sqft"] - df.loc[mask, "calculated_lot_sqft"]).abs()
            / df.loc[mask, "calculated_lot_sqft"]
            * 100
        )

        # Check for significant differences (>5%)
        significant_diff = (df["lot_size_diff_pct"] > 5).sum()
        if significant_diff > 0:
            logger.warning(
                f"Found {significant_diff} properties with significant lot size differences"
            )

        # Clean up temporary columns
        df = df.drop(
            columns=["calculated_lot_sqft", "lot_size_diff_pct"], errors="ignore"
        )

    # Define lot size categories
    logger.info("Categorizing lot sizes")

    def categorize_lot_size(row):
        lot_sqft = row.get("lot_sqft")
        acres = row.get("acres")

        # If lot_sqft is available, use it directly
        if pd.notna(lot_sqft) and lot_sqft > 0:
            if lot_sqft < 3000:
                return "Small Lot"
            elif lot_sqft < 6000:
                return "Standard Lot"
            elif lot_sqft < 12000:
                return "Large Lot"
            elif lot_sqft < 43560:
                return "Estate Lot"
            else:
                return "Acreage"

        # If acres is available, use it as fallback
        elif pd.notna(acres) and acres > 0:
            if acres < 0.07:  # Approx 3,000 sqft
                return "Small Lot"
            elif acres < 0.14:  # Approx 6,000 sqft
                return "Standard Lot"
            elif acres < 0.28:  # Approx 12,000 sqft
                return "Large Lot"
            elif acres < 1:
                return "Estate Lot"
            else:
                return "Acreage"

        # If neither is available
        return "Unknown"

    df["lot_size_category"] = df.apply(categorize_lot_size, axis=1)
    logger.info(
        f"Lot size categories: {df['lot_size_category'].value_counts().to_dict()}"
    )

    return df


def calculate_hoa_dues_yn(df):
    """
    Calculate HOA dues Y/N flag

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with hoa_dues_y_n column
    """
    # Check if HOA dues columns exist
    hoa_fee_col = next(
        (col for col in ["hoa_fee", "association_fee"] if col in df.columns), None
    )
    annual_hoa_col = next(
        (
            col
            for col in ["annual_hoa_fee", "total_hoa_fee_per_year"]
            if col in df.columns
        ),
        None,
    )

    if not (hoa_fee_col or annual_hoa_col):
        logger.warning("No HOA fee columns found, cannot calculate HOA dues Y/N")
        return df

    logger.info("Calculating HOA dues Y/N flag")

    # Initialize hoa_dues_y_n column
    if "hoa_dues_y_n" not in df.columns:
        df["hoa_dues_y_n"] = "No"

    # Calculate based on hoa_fee
    if hoa_fee_col:
        df[hoa_fee_col] = pd.to_numeric(df[hoa_fee_col], errors="coerce").fillna(0)
        mask = df[hoa_fee_col] > 0
        df.loc[mask, "hoa_dues_y_n"] = "Yes"

    # Calculate based on annual_hoa_fee
    if annual_hoa_col:
        df[annual_hoa_col] = pd.to_numeric(df[annual_hoa_col], errors="coerce").fillna(
            0
        )
        mask = df[annual_hoa_col] > 0
        df.loc[mask, "hoa_dues_y_n"] = "Yes"

    hoa_dues_count = (df["hoa_dues_y_n"] == "Yes").sum()
    logger.info(f"Properties with HOA dues: {hoa_dues_count}")

    return df


def determine_str_eligibility(df):
    """
    Determine short-term rental eligibility

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with str_eligible column
    """
    # Check if short term rental column exists
    str_col = next(
        (col for col in ["short_term_rental", "short_termable"] if col in df.columns),
        None,
    )

    if not str_col:
        logger.warning("No short term rental column found, will rely solely on zoning")

    logger.info("Determining short-term rental eligibility")

    # Initialize str_eligible column
    df["str_eligible"] = "No"

    # Define property types that may allow STRs
    str_property_types = ["condo", "condominium", "townhouse", "townhome", "vacation"]

    # Use short term rental column if available
    if str_col and "property_type" in df.columns:
        # Make sure property_type is string type before using string methods
        prop_type_strings = df["property_type"].fillna("").astype(str)
        
        for prop_type in str_property_types:
            mask = prop_type_strings.str.lower().str.contains(prop_type, na=False)
            df.loc[mask, "str_eligible"] = "Yes"

    # Use zoning if available
    if "zoning" in df.columns:
        # Make sure zoning is string type before using string methods
        zoning_strings = df["zoning"].fillna("").astype(str)
        
        # Check for STR-friendly zoning codes
        str_zoning_patterns = ["str", "short term", "resort", "tourist", "vacation"]

        for pattern in str_zoning_patterns:
            mask = zoning_strings.str.lower().str.contains(pattern, na=False)
            df.loc[mask, "str_eligible"] = "Yes"

    # Check for property types that may be STR-eligible
    if "property_type" in df.columns:
        # Make sure property_type is string type before using string methods
        prop_type_strings = df["property_type"].fillna("").astype(str)
        
        for prop_type in str_property_types:
            mask = prop_type_strings.str.lower().str.contains(prop_type, na=False)
            # Mark as 'Potential' if not already 'Yes'
            potential_mask = mask & (df["str_eligible"] != "Yes")
            df.loc[potential_mask, "str_eligible"] = "Potential"

    # Count properties by STR eligibility
    str_counts = df["str_eligible"].value_counts().to_dict()
    logger.info(f"STR eligibility counts: {str_counts}")

    return df


def determine_furnished_status(df):
    """
    Determine furnished status

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with furnished column
    """
    # Check if furnished column exists
    furnished_col = next(
        (col for col in ["is_furnished", "furnished"] if col in df.columns), None
    )

    if not furnished_col:
        logger.warning(
            "No furnished column found, will create one with 'Unknown' values"
        )
        df["furnished"] = "Unknown"
        return df

    logger.info("Determining furnished status")

    # Map values to Yes/No
    yes_values = ["yes", "y", "true", "furnished", "partially"]

    df["furnished"] = df[furnished_col].astype(str).str.lower()
    df["furnished"] = df["furnished"].apply(
        lambda x: "Yes" if any(val in x for val in yes_values) else "No"
    )

    # Count furnished properties
    furnished_count = (df["furnished"] == "Yes").sum()
    logger.info(f"Furnished properties: {furnished_count}")

    return df


def check_recent_remodel(df):
    """
    Check for recently remodeled properties

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with recently_remodeled column
    """
    # Check if year_remodeled column exists
    if "year_remodeled" not in df.columns:
        logger.warning(
            "No year_remodeled column found, cannot determine recent remodels"
        )
        df["recently_remodeled"] = "Unknown"
        return df

    logger.info("Checking for recently remodeled properties")

    # Convert to numeric and handle missing values
    df["year_remodeled"] = pd.to_numeric(df["year_remodeled"], errors="coerce")

    # Define "recent" as within the last 10 years
    current_year = pd.Timestamp.now().year
    remodel_threshold = current_year - 10

    # Flag recent remodels
    mask = (df["year_remodeled"] >= remodel_threshold) & (
        df["year_remodeled"] <= current_year
    )
    df["recently_remodeled"] = "No"
    df.loc[mask, "recently_remodeled"] = "Yes"

    # Count recently remodeled properties
    remodeled_count = (df["recently_remodeled"] == "Yes").sum()
    logger.info(f"Recently remodeled properties: {remodeled_count}")

    return df


def build_full_address(df):
    """
    Build full_address using all required components including unit numbers

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with standardized full_address
    """
    # Map potential column names to standard names
    address_column_map = {
        "street_number": ["street_number", "street #", "street no", "house number"],
        "street_direction_prefix": [
            "street_direction_prefix",
            "street_direction_pfx",
            "direction",
            "dir prefix",
        ],
        "street_name": ["street_name", "street", "st name"],
        "unit_number": [
            "unit_number",
            "address_2_unit",
            "unit #",
            "unit no",
            "apartment",
        ],
        "street_suffix": ["street_suffix", "suffix", "st suffix", "street type"],
        "city": ["city", "city_town", "town"],
        "state": ["state", "state_province", "province"],
        "zip_code": ["zip_code", "zip", "postal_code", "postal code"],
    }

    # Find actual column names in the DataFrame
    address_columns = {}
    for component, possible_names in address_column_map.items():
        for name in possible_names:
            if name in df.columns:
                address_columns[component] = name
                break

    # Check if we have enough components to build an address
    min_components = ["street_number", "street_name"]
    if not all(comp in address_columns for comp in min_components):
        logger.warning(f"Missing minimum required address components: {min_components}")
        return df

    logger.info("Building full_address from components")

    # Create address components with proper handling of missing values
    components = {}
    for component, column in address_columns.items():
        components[component] = df[column].fillna("")

    # Build the address
    # Start with street number and direction
    full_address = components.get("street_number", "")

    # Add direction prefix if available
    if "street_direction_prefix" in components:
        direction = components["street_direction_prefix"]
        # Only add space if direction is not empty
        full_address = full_address + direction.apply(lambda x: f" {x}" if x else "")

    # Add street name
    full_address = full_address + " " + components.get("street_name", "")

    # Add street suffix if available
    if "street_suffix" in components:
        full_address = full_address + " " + components["street_suffix"]

    # Add unit number if available
    if "unit_number" in components:
        unit = components["unit_number"]
        full_address = full_address + unit.apply(
            lambda x: f" #{x}" if x and str(x).strip() else ""
        )

    # Add city, state, zip
    city = components.get("city", pd.Series(["Aspen"] * len(df)))
    state = components.get("state", pd.Series(["CO"] * len(df)))
    zip_code = components.get("zip_code", pd.Series(["81611"] * len(df)))

    full_address = (
        full_address + ", " + city + ", " + state + " " + zip_code.astype(str)
    )

    # Clean up any double spaces and trailing/leading spaces
    full_address = full_address.str.replace(r"\s+", " ", regex=True)
    full_address = full_address.str.replace(r",\s*,", ",", regex=True)
    full_address = full_address.str.strip()

    # Add to DataFrame
    df["full_address"] = full_address

    logger.info("Full address generation complete")
    return df


def apply_all_property_features(df):
    """
    Apply all property feature transformations

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with all property features
    """
    logger.info("Applying all property feature transformations")

    # Extract property condition
    df = extract_property_condition(df)

    # Calculate bath score
    df = calculate_bath_score(df)

    # Calculate below ground metrics
    df = calculate_below_ground_metrics(df)

    # Calculate lot size metrics
    df = calculate_lot_size_metrics(df)

    # Calculate HOA dues Y/N
    df = calculate_hoa_dues_yn(df)

    # Determine STR eligibility
    df = determine_str_eligibility(df)

    # Determine furnished status
    df = determine_furnished_status(df)

    # Check for recent remodels
    df = check_recent_remodel(df)

    # Build full address
    df = build_full_address(df)

    logger.info("All property feature transformations complete")
    return df


def main(input_file, output_file=None):
    """Main function to extract property features from a CSV file"""
    logger.info(f"Extracting property features from: {input_file}")

    # Determine output file if not specified
    if output_file is None:
        base_name = os.path.basename(input_file)
        file_name, ext = os.path.splitext(base_name)
        output_file = f"{file_name}_features{ext}"

    # Load the data
    try:
        df = pd.read_csv(input_file, low_memory=False)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except UnicodeDecodeError:
        # Try with different encoding
        df = pd.read_csv(input_file, low_memory=False, encoding="latin-1")
        logger.info(
            f"Loaded {len(df)} rows with {len(df.columns)} columns using latin-1 encoding"
        )
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return False

    # Apply all property features
    df = apply_all_property_features(df)

    # Save processed file
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to: {output_file}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract property features from real estate data"
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output", "-o", help="Path to output CSV file")

    args = parser.parse_args()

    main(args.input_file, args.output)

Can you update the determine_str_eligibility section with this: 

def determine_str_eligibility(df):
    """
    Determine short-term rental eligibility

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with str_eligible column
    """
    # Check if short term rental column exists
    str_col = next(
        (col for col in ["short_term_rental", "short_termable"] if col in df.columns),
        None,
    )

    if not str_col:
        logger.warning("No short term rental column found, will rely solely on zoning")

    logger.info("Determining short-term rental eligibility")

    # Initialize str_eligible column
    df["str_eligible"] = "No"

    # Define property types that may allow STRs
    str_property_types = ["condo", "condominium", "townhouse", "townhome", "vacation"]

    # Use short term rental column if available
    if str_col and "property_type" in df.columns:
        # Make sure property_type is string type before using string methods
        prop_type_strings = df["property_type"].fillna("").astype(str)
        
        for prop_type in str_property_types:
            mask = prop_type_strings.str.lower().str.contains(prop_type, na=False)
            df.loc[mask, "str_eligible"] = "Yes"

    # Use zoning if available
    if "zoning" in df.columns:
        # Make sure zoning is string type before using string methods
        zoning_strings = df["zoning"].fillna("").astype(str)
        
        # Check for STR-friendly zoning codes
        str_zoning_patterns = ["str", "short term", "resort", "tourist", "vacation"]

        for pattern in str_zoning_patterns:
            mask = zoning_strings.str.lower().str.contains(pattern, na=False)
            df.loc[mask, "str_eligible"] = "Yes"

    # Check for property types that may be STR-eligible
    if "property_type" in df.columns:
        # Make sure property_type is string type before using string methods
        prop_type_strings = df["property_type"].fillna("").astype(str)
        
        for prop_type in str_property_types:
            mask = prop_type_strings.str.lower().str.contains(prop_type, na=False)
            # Mark as 'Potential' if not already 'Yes'
            potential_mask = mask & (df["str_eligible"] != "Yes")
            df.loc[potential_mask, "str_eligible"] = "Potential"

    # Count properties by STR eligibility
    str_counts = df["str_eligible"].value_counts().to_dict()
    logger.info(f"STR eligibility counts: {str_counts}")

    return df

And send me the full updated code back?
