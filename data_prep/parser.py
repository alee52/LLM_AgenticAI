from data_prep.items import Item
import json
import re
from PIL import Image

MIN_CHARS = 1
MAX_TEXT_EACH = 2500
MAX_TEXT_TOTAL = 2750

def top_level_category_extractor(full_category):
    #extract the top level category from the full category string, which is in the format "Top Level > Subcategory 1 > Subcategory 2"
    top_lev = full_category.split(' > ')[0]
    return top_lev

def simplify(text_list) -> str:
    """
    Return a simplified string without too much whitespace and limited to MAX_TEXT characters
    """
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title, description) -> str:
    """
    Return a cleansed full string with product numbers and unimportant details removed.
    """
    title = simplify(title) if title else ""
    description = simplify(description) if description else ""

    result = title + "\n"
    if description:
        result += description + "\n"

    # Removes product/barcode-like codes:
    # HP1044, ABC123456, 123456789012
    pattern = r"\b(?:(?=[A-Z0-9]{6,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+|\d{8,14})\b"

    cleaned = re.sub(pattern, "", result)

    # Remove extra spaces left behind
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()[:MAX_TEXT_TOTAL]

def parse(datapoint):
    ground_truth_category = datapoint["ground_truth_category"]
    product_description = datapoint["product_description"]
    product_title = datapoint["product_title"]

    if not all(isinstance(x, str) for x in [
        ground_truth_category,
        product_description,
        product_title,
    ]):
        return None

    if len(product_title) < MIN_CHARS:
        return None

    if len(product_description) < MIN_CHARS:
        return None
    
    # if datapoint["image_url"] is None:
    #     return None

    full = scrub(product_title, product_description)

    return Item(
        category=top_level_category_extractor(ground_truth_category),
        full=full,
        # image=datapoint["product_image"],  # Initialize image with the URL
    )




