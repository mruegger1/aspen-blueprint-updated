"""
Address matching functionality for real estate property lookup.
"""

import re
import fuzzywuzzy.fuzz as fuzz
from utils import logger


class AddressMatcher:
    """Handles address matching and property lookup."""

    @staticmethod
    def find_property_by_address(data, address_query, threshold=80):
        """
        Find a property by address using fuzzy matching.

        Args:
            data (DataFrame): The property data
            address_query (str): The address to search for
            threshold (int): Minimum match score (0-100)

        Returns:
            DataFrame: The matching property row or None if not found
        """
        if "full_address" not in data.columns:
            # Try to create full_address from components
            address_cols = [col for col in data.columns if "address" in col.lower()]
            if not address_cols:
                logger.error("No address columns found in the data")
                return None

            # Use the first address column found
            address_col = address_cols[0]
            logger.info(f"Using {address_col} for address matching")
            addresses = data[address_col].fillna("")
        else:
            addresses = data["full_address"].fillna("")

        # Create a list of addresses for matching
        address_list = list(addresses)

        # Normalize addresses
        clean_addresses = [
            AddressMatcher.normalize_address(addr) for addr in address_list
        ]
        clean_query = AddressMatcher.normalize_address(address_query)

        # Find the best match using a direct approach
        best_match_idx = -1
        best_match_score = -1

        for i, clean_addr in enumerate(clean_addresses):
            score = fuzz.token_set_ratio(clean_query, clean_addr)
            if score > best_match_score and score >= threshold:
                best_match_score = score
                best_match_idx = i

        if best_match_idx == -1:
            logger.warning(f"No matches found for address: {address_query}")
            return None

        # Log the best match
        logger.info(
            f"Best match: {address_list[best_match_idx]} (Score: {best_match_score})"
        )

        # Find other good matches for logging
        other_matches = []
        for i, clean_addr in enumerate(clean_addresses):
            if i != best_match_idx:
                score = fuzz.token_set_ratio(clean_query, clean_addr)
                if score >= threshold:
                    other_matches.append((i, score))

        # Sort other matches by score (descending)
        other_matches.sort(key=lambda x: x[1], reverse=True)

        # Log other matches
        if other_matches:
            logger.info("Other potential matches:")
            for i, score in other_matches[:4]:  # Show top 4 other matches
                logger.info(f"  {address_list[i]} (Score: {score})")

        # Return the property data
        return data.iloc[best_match_idx : best_match_idx + 1]

    @staticmethod
    def normalize_address(address):
        """
        Normalize an address string for better matching.

        Args:
            address (str): The address to normalize

        Returns:
            str: The normalized address
        """
        if not isinstance(address, str):
            return ""

        # Convert to lowercase
        norm = address.lower()

        # Replace common abbreviations
        replacements = {
            "avenue": "ave",
            "street": "st",
            "road": "rd",
            "boulevard": "blvd",
            "drive": "dr",
            "lane": "ln",
            "circle": "cir",
            "court": "ct",
            "highway": "hwy",
            "apartment": "apt",
            "suite": "ste",
            "unit": "#",
        }

        for full, abbr in replacements.items():
            norm = norm.replace(f" {full} ", f" {abbr} ")
            norm = norm.replace(f" {full}, ", f" {abbr}, ")
            norm = norm.replace(f" {full}.", f" {abbr}.")

        # Remove punctuation except for # sign
        norm = re.sub(r"[^\w\s#]", "", norm)

        # Remove extra whitespace
        norm = re.sub(r"\s+", " ", norm).strip()

        return norm
