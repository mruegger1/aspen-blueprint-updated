#!/usr/bin/env python3
"""
Run Classic Comps Menu - Interactive CLI for Traditional Comp Finder
-------------------------------------------------------------------
Provides a user-friendly menu interface for the classic comp finder.
"""

import sys
import os
import datetime
import argparse

# Add the parent directory to the path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import the core functionality
from comp_finder_classic import run_classic_comp_search
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
    path = os.path.expanduser(file_path)

    if os.path.isfile(path):
        return os.path.abspath(path)

    # Not found
    raise ValueError(f"File not found: {file_path}")


def run_interactive_menu():
    """Run the interactive classic property analysis menu."""
    print("\n" + "=" * 80)
    print("ASPEN CLASSIC COMP FINDER MENU")
    print("=" * 80)
    print(
        "\nThis tool helps you find comparable properties using traditional criteria."
    )

    # Create outputs directory if it doesn't exist
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Created output directory: {outputs_dir}")

    # Ask for CSV path
    csv_path_input = get_user_input(
        "Enter the path to your CSV file (leave empty for default)"
    )

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
        print("1. Find Comps by Criteria")
        print("2. Find Comps Similar to a Property")
        print("3. Exit")

        choice = get_user_input("Enter your choice", "1")

        if choice == "1":
            run_criteria_search_menu(outputs_dir, csv_path)
        elif choice == "2":
            run_similar_property_menu(outputs_dir, csv_path)
        elif choice == "3":
            print("\nThank you for using the Aspen Classic Comp Finder!")
            break
        else:
            print("Invalid option, please try again.")


def run_criteria_search_menu(outputs_dir, csv_path=None):
    """Run the classic comp finder by criteria workflow."""
    print("\n" + "-" * 80)
    print("Find Comps by Criteria")
    print("-" * 80)

    # Get basic search parameters
    bedrooms = get_user_input(
        "Number of bedrooms (leave empty to skip)",
        None,
        lambda x: validate_number(x, min_val=0, max_val=20, is_int=True) if x else None,
    )

    bathrooms = get_user_input(
        "Number of bathrooms (leave empty to skip)",
        None,
        lambda x: validate_number(x, min_val=0, max_val=20) if x else None,
    )

    property_type = get_user_input(
        "Property type (e.g. Condo, Townhouse, Single Family)", None
    )

    area = get_user_input("Area or neighborhood", None)

    str_eligible_input = get_user_input(
        "Short-term rental eligible (y/n, leave empty to skip)", None
    )
    str_eligible = None
    if str_eligible_input:
        str_eligible = validate_yes_no(str_eligible_input)

    condition = get_user_input(
        "Property condition (e.g. Excellent, Good, Average)", None
    )

    max_price_input = get_user_input("Maximum price (leave empty to skip)", None)
    max_price = None
    if max_price_input:
        max_price = validate_number(max_price_input, min_val=0)

    sqft_min_input = get_user_input(
        "Minimum square footage (leave empty to skip)", None
    )
    sqft_min = None
    if sqft_min_input:
        sqft_min = validate_number(sqft_min_input, min_val=0)

    sqft_max_input = get_user_input(
        "Maximum square footage (leave empty to skip)", None
    )
    sqft_max = None
    if sqft_max_input:
        sqft_max = validate_number(sqft_max_input, min_val=0)

    status_options = {
        "A": "Active listings only",
        "P": "Pending listings only",
        "S": "Sold properties only",
        "": "All statuses",
    }

    print("\nListing Status Options:")
    for key, desc in status_options.items():
        print(f"  {key or 'Enter'}: {desc}")

    listing_status = get_user_input("Select listing status", "S")
    if not listing_status:
        listing_status = None

    months_back = get_user_input(
        "For sold properties, how many months back to search",
        "24",
        lambda x: validate_number(x, min_val=1, max_val=60, is_int=True),
    )

    limit = get_user_input(
        "Number of comps to return",
        "5",
        lambda x: validate_number(x, min_val=1, max_val=20, is_int=True),
    )

    export = get_user_input("Export results to CSV (y/n)", "y", validate_yes_no)

    # Confirmation
    print("\nSearch parameters:")
    if bedrooms is not None:
        print(f"  Bedrooms: {bedrooms}")
    if bathrooms is not None:
        print(f"  Bathrooms: {bathrooms}")
    if property_type:
        print(f"  Property Type: {property_type}")
    if area:
        print(f"  Area: {area}")
    if str_eligible is not None:
        print(f"  STR Eligible: {'Yes' if str_eligible else 'No'}")
    if condition:
        print(f"  Condition: {condition}")
    if max_price:
        print(f"  Max Price: ${max_price:,.0f}")
    if sqft_min:
        print(f"  Min Sqft: {sqft_min}")
    if sqft_max:
        print(f"  Max Sqft: {sqft_max}")

    status_text = status_options.get(listing_status, "All statuses")
    print(f"  Status: {status_text}")

    if listing_status in [None, "S", "Sold", "SOLD"]:
        print(f"  Months Back: {months_back}")

    print(f"  Number of Comps: {limit}")

    confirm = get_user_input("Proceed with search (y/n)", "y", validate_yes_no)
    if not confirm:
        print("Search cancelled.")
        return

    # Create a timestamped subfolder for this analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(outputs_dir, f"classic_comps_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    print("\nSearching for comparable properties...")
    print("Please wait, this may take a moment...")

    try:
        # Run the search
        run_classic_comp_search(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            property_type=property_type,
            area=area,
            str_eligible=str_eligible,
            condition=condition,
            max_price=max_price,
            sqft_min=sqft_min,
            sqft_max=sqft_max,
            months_back=int(months_back),
            listing_status=listing_status,
            limit=int(limit),
            csv_path=csv_path,
            export_results=export,
            export_dir=analysis_dir,
        )

    except Exception as e:
        print(f"\nError during search: {str(e)}")
        import traceback

        traceback.print_exc()


def run_similar_property_menu(outputs_dir, csv_path=None):
    """Run the similar property workflow."""
    print("\n" + "-" * 80)
    print("Find Comps Similar to a Property")
    print("-" * 80)

    # Get property address
    address = get_user_input("Enter the property address to find similar properties")
    if not address:
        print("Operation cancelled.")
        return

    # Get refinement options
    print("\nRefinement Options (leave empty to use reference property's attributes):")

    bedrooms = get_user_input(
        "Number of bedrooms (leave empty to use reference property)",
        None,
        lambda x: validate_number(x, min_val=0, max_val=20, is_int=True) if x else None,
    )

    bathrooms = get_user_input(
        "Number of bathrooms (leave empty to use reference property)",
        None,
        lambda x: validate_number(x, min_val=0, max_val=20) if x else None,
    )

    status_options = {
        "A": "Active listings only",
        "P": "Pending listings only",
        "S": "Sold properties only",
        "": "All statuses",
    }

    print("\nListing Status Options:")
    for key, desc in status_options.items():
        print(f"  {key or 'Enter'}: {desc}")

    listing_status = get_user_input("Select listing status", "S")
    if not listing_status:
        listing_status = None

    months_back = get_user_input(
        "For sold properties, how many months back to search",
        "24",
        lambda x: validate_number(x, min_val=1, max_val=60, is_int=True),
    )

    limit = get_user_input(
        "Number of comps to return",
        "5",
        lambda x: validate_number(x, min_val=1, max_val=20, is_int=True),
    )

    export = get_user_input("Export results to CSV (y/n)", "y", validate_yes_no)

    # Create a timestamped subfolder for this analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(outputs_dir, f"property_comps_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    # Confirmation
    print(f"\nFinding properties similar to: {address}")

    status_text = status_options.get(listing_status, "All statuses")
    print(f"Status: {status_text}")

    if listing_status in [None, "S", "Sold", "SOLD"]:
        print(f"Months Back: {months_back}")

    print(f"Number of Comps: {limit}")

    print("\nSearching for comparable properties...")
    print("Please wait, this may take a moment...")

    try:
        # Run the search
        run_classic_comp_search(
            reference_property_address=address,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            months_back=int(months_back),
            listing_status=listing_status,
            limit=int(limit),
            csv_path=csv_path,
            export_results=export,
            export_dir=analysis_dir,
        )

    except Exception as e:
        print(f"\nError during search: {str(e)}")
        import traceback

        traceback.print_exc()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aspen Classic Comp Finder")

    # Property criteria
    parser.add_argument(
        "--address", type=str, help="Find comps similar to this address"
    )
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=float, help="Number of bathrooms")
    parser.add_argument(
        "--property-type", type=str, help="Property type (e.g. Condo, Single Family)"
    )
    parser.add_argument("--area", type=str, help="Area or neighborhood")
    parser.add_argument(
        "--str-eligible", type=str, help="Short-term rental eligible (yes/no)"
    )
    parser.add_argument("--condition", type=str, help="Property condition")
    parser.add_argument("--max-price", type=float, help="Maximum price")
    parser.add_argument("--sqft-min", type=float, help="Minimum square footage")
    parser.add_argument("--sqft-max", type=float, help="Maximum square footage")
    parser.add_argument("--year-built-min", type=int, help="Minimum year built")

    # Search options
    parser.add_argument(
        "--listing-status",
        type=str,
        choices=["A", "P", "S"],
        default="S",
        help="Listing status (A=Active, P=Pending, S=Sold)",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=24,
        help="For sold properties, months to look back",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of comps to return"
    )
    parser.add_argument("--csv", type=str, help="Path to CSV file with property data")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory for output files"
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Don't export results to CSV"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Create outputs directory
    outputs_dir = args.output_dir
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # If address is provided, find comps similar to that property
    if args.address:
        print(f"Finding comps similar to: {args.address}")

        # Process boolean parameters
        str_eligible = None
        if args.str_eligible:
            str_eligible = args.str_eligible.lower() in ["yes", "y", "true", "1"]

        try:
            run_classic_comp_search(
                reference_property_address=args.address,
                bedrooms=args.bedrooms,
                bathrooms=args.bathrooms,
                property_type=args.property_type,
                area=args.area,
                str_eligible=str_eligible,
                condition=args.condition,
                max_price=args.max_price,
                sqft_min=args.sqft_min,
                sqft_max=args.sqft_max,
                year_built_min=args.year_built_min,
                months_back=args.months_back,
                listing_status=args.listing_status,
                limit=args.limit,
                csv_path=args.csv,
                export_results=not args.no_export,
                export_dir=outputs_dir,
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()

    # Else if any criteria are provided, use them for search
    elif any(
        [
            args.bedrooms,
            args.bathrooms,
            args.property_type,
            args.area,
            args.str_eligible,
            args.condition,
            args.max_price,
            args.sqft_min,
            args.sqft_max,
            args.year_built_min,
        ]
    ):
        # Process boolean parameters
        str_eligible = None
        if args.str_eligible:
            str_eligible = args.str_eligible.lower() in ["yes", "y", "true", "1"]

        try:
            run_classic_comp_search(
                bedrooms=args.bedrooms,
                bathrooms=args.bathrooms,
                property_type=args.property_type,
                area=args.area,
                str_eligible=str_eligible,
                condition=args.condition,
                max_price=args.max_price,
                sqft_min=args.sqft_min,
                sqft_max=args.sqft_max,
                year_built_min=args.year_built_min,
                months_back=args.months_back,
                listing_status=args.listing_status,
                limit=args.limit,
                csv_path=args.csv,
                export_results=not args.no_export,
                export_dir=outputs_dir,
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()

    # Otherwise run interactive menu
    else:
        run_interactive_menu()


if __name__ == "__main__":
    main()
