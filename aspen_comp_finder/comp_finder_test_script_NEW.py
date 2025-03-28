import pandas as pd
import numpy as np
import re
import sys
import matplotlib.pyplot as plt

# Predefined building and street premiums
building_premiums = {
    "Monarch on the park": 0.604,
    "DurCondo": 0.48,
    "MtQueen": 0.424,
    "DurMews": 0.343,
    "TwnSteAs": 0.267,
    "NorthNell": 0.172,
    "DerBerghof": 0.146,
    "Dolomite": 0.128,
    "Aspen Alps": 0.106,
    "Aspen Square": 0.085,
    "ObermeyerPlace": 0.068,
    "FifthAve": 0.063,
    "ShadowMtn": 0.052,
    "210Cooper": 0.04,
    "SouthPt": 0.029,
}

street_premiums = {
    "galena": 0.271,
    "monarch": 0.186,
    "durant": 0.032,
    "mill": 0.025,
    "cooper": 0.02,
    "hyman": -0.016,
    "aspen": -0.059,
    "hopkins": -0.075,
    "main": -0.203,
}


def extract_street_name(address):
    """
    Extract street name from a full address with precise filtering.

    Args:
        address (str): Full address string

    Returns:
        str: Extracted street name in lowercase
    """
    if pd.isna(address):
        return None

    # Remove unit/apartment numbers
    address = re.sub(r"\s*(?:unit|#)\s*\d+", "", address, flags=re.IGNORECASE)

    # Split the address and extract the street name
    parts = address.split()
    if len(parts) > 1:
        # Take the second part (assuming street name starts after street number)
        street_name = " ".join(
            parts[1:2]
        ).lower()  # Only take the first word after street number

        # Remove street suffixes
        street_suffixes = [
            "street",
            "st",
            "avenue",
            "ave",
            "road",
            "rd",
            "lane",
            "ln",
            "drive",
            "dr",
            "way",
            "court",
            "ct",
        ]
        for suffix in street_suffixes:
            street_name = street_name.replace(suffix, "").strip()

        # Exclude common city/town names
        city_names = ["aspen", "snowmass", "basalt"]
        if street_name in city_names:
            return None

        return street_name

    return None


def analyze_dataset(csv_path):
    """
    Perform comprehensive analysis on the dataset.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        dict: Detailed analysis results
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Analysis dictionary
    analysis = {
        "dataset_info": {"total_rows": len(df), "columns": list(df.columns)},
        "building_premium_analysis": {
            "buildings_found": {},
            "total_properties_with_building_premium": 0,
        },
        "street_premium_analysis": {
            "streets_found": {},
            "total_properties_with_street_premium": 0,
        },
    }

    # Analyze building premiums
    if "sub_loc" in df.columns:
        for building in building_premiums.keys():
            building_matches = df[
                df["sub_loc"].str.contains(building, case=False, na=False)
            ]
            if len(building_matches) > 0:
                analysis["building_premium_analysis"]["buildings_found"][building] = (
                    len(building_matches)
                )
                analysis["building_premium_analysis"][
                    "total_properties_with_building_premium"
                ] += len(building_matches)

    # Analyze street premiums
    if "full_address" in df.columns:
        df["street_name"] = df["full_address"].apply(extract_street_name)

        for street in street_premiums.keys():
            street_matches = df[df["street_name"] == street]
            if len(street_matches) > 0:
                analysis["street_premium_analysis"]["streets_found"][street] = len(
                    street_matches
                )
                analysis["street_premium_analysis"][
                    "total_properties_with_street_premium"
                ] += len(street_matches)

    # Additional property insights
    analysis["property_insights"] = {
        "property_types": df["resolved_property_type"].value_counts().to_dict(),
        "areas": df["area"].value_counts().to_dict(),
        "demo_score_distribution": df["demo_score"].describe().to_dict(),
        "str_eligibility": df["str_eligible"].value_counts().to_dict(),
    }

    return analysis


def visualize_insights(df):
    """
    Create visualizations for key insights.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Set up the plotting configuration
    plt.rcParams.update({"font.size": 10, "figure.figsize": (12, 5), "axes.grid": True})

    # Demo Score Distribution
    plt.figure(figsize=(12, 5))

    # Subplot 1: Demo Score Histogram
    plt.subplot(1, 2, 1)
    df["demo_score"].hist(bins=20, edgecolor="black")
    plt.title("Demo Score Distribution")
    plt.xlabel("Demo Score")
    plt.ylabel("Frequency")

    # Subplot 2: Demo Score by Area
    plt.subplot(1, 2, 2)
    demo_by_area = df.groupby("area")["demo_score"].mean().sort_values(ascending=False)
    demo_by_area.plot(kind="bar")
    plt.title("Average Demo Score by Area")
    plt.xlabel("Area")
    plt.ylabel("Average Demo Score")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("demo_score_analysis.png")
    plt.close()

    # STR Eligibility by Property Type
    plt.figure(figsize=(10, 6))
    str_by_type = (
        df.groupby(["resolved_property_type", "str_eligible"])
        .size()
        .unstack(fill_value=0)
    )
    str_by_type.plot(kind="bar", stacked=True)
    plt.title("STR Eligibility by Property Type")
    plt.xlabel("Property Type")
    plt.ylabel("Number of Properties")
    plt.legend(title="STR Eligible", loc="upper right")
    plt.tight_layout()
    plt.savefig("str_eligibility_by_type.png")
    plt.close()


def print_analysis_results(analysis):
    """
    Print comprehensive analysis results.

    Args:
        analysis (dict): Analysis results dictionary
    """
    print("\n=== DATASET ANALYSIS ===")
    print(f"Total Properties: {analysis['dataset_info']['total_rows']}")

    print("\n=== BUILDING PREMIUMS ===")
    print("Buildings Found:")
    for building, count in analysis["building_premium_analysis"][
        "buildings_found"
    ].items():
        print(f"  {building}: {count} properties")
    print(
        f"Total Properties with Building Premiums: {analysis['building_premium_analysis']['total_properties_with_building_premium']}"
    )

    print("\n=== STREET PREMIUMS ===")
    print("Streets Found:")
    for street, count in analysis["street_premium_analysis"]["streets_found"].items():
        print(f"  {street}: {count} properties")
    print(
        f"Total Properties with Street Premiums: {analysis['street_premium_analysis']['total_properties_with_street_premium']}"
    )

    print("\n=== PROPERTY INSIGHTS ===")
    print("Property Types:")
    for pt, count in analysis["property_insights"]["property_types"].items():
        print(f"  {pt}: {count}")

    print("\nAreas:")
    for area, count in analysis["property_insights"]["areas"].items():
        print(f"  {area}: {count}")

    print("\nDemo Score Distribution:")
    demo_stats = analysis["property_insights"]["demo_score_distribution"]
    for stat, value in demo_stats.items():
        print(f"  {stat}: {value}")

    print("\nSTR Eligibility:")
    for eligibility, count in analysis["property_insights"]["str_eligibility"].items():
        print(f"  {eligibility}: {count}")


def main():
    # Check if CSV path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide the path to the CSV file as a command-line argument.")
        sys.exit(1)

    # Path to your CSV file from command-line argument
    csv_path = sys.argv[1]

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Run comprehensive analysis
    analysis_results = analyze_dataset(csv_path)

    # Print analysis results
    print_analysis_results(analysis_results)

    # Create visualizations
    visualize_insights(df)

    print("\n=== VISUALIZATIONS ===")
    print("Generated visualizations:")
    print("  1. demo_score_analysis.png")
    print("  2. str_eligibility_by_type.png")


if __name__ == "__main__":
    main()
