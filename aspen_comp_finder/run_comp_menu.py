#!/usr/bin/env python3
"""
Run Comp Menu - Interactive CLI for Aspen Real Estate Comp Finder
-----------------------------------------------------------------
Provides a user-friendly menu interface for the property analysis tools.
"""

import sys
import os
import datetime
import argparse
import glob
from pathlib import Path

# Add the parent directory to the path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import the core functionality
from comp_finder import run_comp_analysis_by_address
from utils import logger


def get_user_input(prompt, default=None, validate_func=None):
    """Get user input with validation and default value."""
    prompt_text = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "

    while True:
        user_input = input(prompt_text)

        # Use default if empty
        if not user_input and default is not None:
            return default

        # Validate if needed
        if validate_func and user_input:
            try:
                validated = validate_func(user_input)
                return validated
            except Exception as e:
                print(f"Invalid input: {str(e)}")
                continue

        # Return as is if no validation needed
        if user_input:
            return user_input

        # If we get here, input was empty and no default
        print("Input required.")


def validate_number(value, min_val=None, max_val=None, is_int=False):
    """Validate numeric input."""
    try:
        if is_int:
            result = int(value)
        else:
            result = float(value)

        if min_val is not None and result < min_val:
            raise ValueError(f"Value must be at least {min_val}")

        if max_val is not None and result > max_val:
            raise ValueError(f"Value must be at most {max_val}")

        return result
    except ValueError:
        raise ValueError(f"Please enter a valid {'integer' if is_int else 'number'}")


def validate_yes_no(value):
    """Validate yes/no input."""
    value = value.lower()
    if value in ("y", "yes"):
        return True
    elif value in ("n", "no"):
        return False
    else:
        raise ValueError("Please enter 'y' or 'n'")


def validate_file_path(file_path):
    """Validate that a file exists and return its absolute path."""
    if not file_path:
        return None
        
    # Convert to Path object for better path handling
    path = Path(file_path)
    
    if path.is_file():
        return str(path.absolute())
    
    # If not found directly, try to search for it
    parts = path.parts
    filename = parts[-1]
    
    # Try to find the file recursively from current directory
    search_pattern = f"**/{filename}"
    matches = list(Path('.').glob(search_pattern))
    
    if matches:
        # Found at least one match
        if len(matches) == 1:
            print(f"Found file at: {matches[0]}")
            return str(matches[0].absolute())
        else:
            print(f"Found multiple matching files. Please select one:")
            for i, match in enumerate(matches):
                print(f"{i+1}. {match}")
            
            choice = get_user_input("Enter number", "1", 
                lambda x: validate_number(x, min_val=1, max_val=len(matches), is_int=True))
            
            return str(matches[int(choice)-1].absolute())
    
    # Not found
    possible_locations = []
    
    # Check some common relative paths
    common_paths = [
        "../../Data/mvp",
        "../Data/mvp", 
        "../../Python Analysis/Data/mvp",
        "../Python Analysis/Data/mvp",
        "~/Desktop/Python Analysis/Data/mvp"
    ]
    
    for base_path in common_paths:
        expanded_path = os.path.expanduser(base_path)
        if os.path.exists(expanded_path):
            possible_files = glob.glob(os.path.join(expanded_path, f"*{filename}*"))
            possible_locations.extend(possible_files)
    
    if possible_locations:
        print(f"File '{filename}' not found at specified path.")
        print("However, I found some similar files that might be what you're looking for:")
        for i, loc in enumerate(possible_locations):
            print(f"{i+1}. {loc}")
        
        choice = get_user_input("Enter number (or 0 to cancel)", "0", 
            lambda x: validate_number(x, min_val=0, max_val=len(possible_locations), is_int=True))
        
        if choice == 0:
            raise ValueError(f"File not found: {file_path}")
        
        return possible_locations[choice-1]
    
    # List current directory to help debugging
    print(f"File not found: {file_path}")
    print(f"Current directory: {os.getcwd()}")
    print("Files in current directory:")
    for item in os.listdir('.'):
        print(f"  {item}")
        
    raise ValueError(f"File not found: {file_path}")


def run_interactive_menu():
    """Run the interactive property analysis menu."""
    print("\n" + "=" * 80)
    print("ASPEN PROPERTY ANALYSIS MENU")
    print("=" * 80)
    print("\nThis tool helps you analyze properties and find comparables in Aspen.")

    # Create outputs directory if it doesn't exist
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Created output directory: {outputs_dir}")

    # Ask for CSV path
    csv_path_input = get_user_input("Enter the path to your CSV file (leave empty for default)")
    
    # Validate CSV path if provided
    csv_path = None
    if csv_path_input:
        try:
            csv_path = validate_file_path(csv_path_input)
            print(f"Using CSV file: {csv_path}")
        except ValueError as e:
            print(f"Warning: {str(e)}")
            print("Continuing with default CSV file.")

    while True:
        print("\n" + "-" * 80)
        print("Main Menu")
        print("-" * 80)
        print("1. Analyze Property by Address")
        print("2. Exit")

        choice = get_user_input("Enter your choice", "1")

        if choice == "1":
            run_property_analysis_menu(outputs_dir, csv_path)
        elif choice == "2":
            print("\nThank you for using the Aspen Property Analysis Tool!")
            break
        else:
            print("Invalid option, please try again.")


def run_property_analysis_menu(outputs_dir, csv_path=None):
    """Run the property analysis workflow."""
    print("\n" + "-" * 80)
    print("Property Analysis by Address")
    print("-" * 80)

    # Get address
    address = get_user_input("Enter the property address")
    if not address:
        print("Operation cancelled.")
        return

    # Get analysis parameters with defaults
    print("\nAnalysis Parameters (press Enter to use defaults):")

    limit = get_user_input(
        "Max number of comps to return",
        "5",
        lambda x: validate_number(x, min_val=1, max_val=20, is_int=True),
    )

    min_comps = get_user_input(
        "Minimum number of comps to find",
        "3",
        lambda x: validate_number(x, min_val=1, max_val=limit, is_int=True),
    )

    similarity = get_user_input(
        "Similarity threshold (0-1, higher = stricter)",
        "0.7",
        lambda x: validate_number(x, min_val=0, max_val=1),
    )

    price_diff = get_user_input(
        "Maximum price difference (as percentage 0-1)",
        "0.35",
        lambda x: validate_number(x, min_val=0, max_val=1),
    )

    sqft_diff = get_user_input(
        "Maximum square footage difference",
        "500",
        lambda x: validate_number(x, min_val=0, is_int=True),
    )

    export = get_user_input("Export results to CSV (y/n)", "y", validate_yes_no)

    # Create a timestamped subfolder for this analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(outputs_dir, f"analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\nRunning analysis for: {address}")
    print("Please wait, this may take a moment...")

    try:
        # Run the analysis
        result = run_comp_analysis_by_address(
            address=address,
            limit=limit,
            min_comps=min_comps,
            similarity_threshold=similarity,
            max_price_diff_pct=price_diff,
            max_sqft_diff=sqft_diff,
            export_results=export,
            export_dir=analysis_dir,
            csv_path=csv_path,  # Add the CSV path parameter
        )

        # Analysis output is handled by the function itself

        # Show relaxation summary if applicable
        if "filter_relaxation_applied" in result:
            relaxed = result["filter_relaxation_applied"]
            if relaxed:
                print("\n" + "-" * 80)
                print("FILTER RELAXATION SUMMARY")
                print("-" * 80)
                print(
                    f"Filters were automatically relaxed to find at least {min_comps} comparable properties."
                )
                print("Relaxed filters: " + ", ".join(relaxed))

        print(f"\nResults saved to: {analysis_dir}")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aspen Property Analysis Menu")
    parser.add_argument(
        "--address", type=str, help="Run direct analysis for this address"
    )
    parser.add_argument(
        "--csv", type=str, help="Path to CSV file with property data"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of comps to return"
    )
    parser.add_argument(
        "--min-comps",
        type=int,
        default=3,
        help="Minimum number of comps to try to find",
    )
    parser.add_argument(
        "--similarity", type=float, default=0.7, help="Similarity threshold (0-1)"
    )
    parser.add_argument(
        "--price-diff",
        type=float,
        default=0.35,
        help="Maximum price difference percentage (0-1)",
    )
    parser.add_argument(
        "--sqft-diff", type=int, default=500, help="Maximum square footage difference"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory for output files"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Create outputs directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # If address is provided, run direct analysis
    if args.address:
        print(f"Running direct analysis for: {args.address}")
        
        # Validate CSV path if provided
        csv_path = None
        if args.csv:
            try:
                csv_path = validate_file_path(args.csv)
                print(f"Using CSV file: {csv_path}")
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Please check the file path and try again.")
                return
        
        run_comp_analysis_by_address(
            address=args.address,
            limit=args.limit,
            min_comps=args.min_comps,
            similarity_threshold=args.similarity,
            max_price_diff_pct=args.price_diff,
            max_sqft_diff=args.sqft_diff,
            export_results=True,
            export_dir=args.output_dir,
            csv_path=csv_path,
        )
    else:
        # Otherwise run interactive menu
        run_interactive_menu()


if __name__ == "__main__":
    main()