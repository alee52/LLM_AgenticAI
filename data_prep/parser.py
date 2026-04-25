from data_prep.items import Item
import json
import re

MIN_CHARS = 1
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

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
    Return a cleansed full string with product numbers and unimportant details removed
    """
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]

def parse(datapoint):
    try:
        ground_truth_category = str(datapoint["ground_truth_category"])
    except ValueError:
        return None
    try:
        product_description = str(datapoint["product_description"])
    except ValueError:
        return None
    try:
        product_title = str(datapoint["product_title"])
    except ValueError:
        return None
    
    if MIN_CHARS <= len(product_title) <= MAX_TEXT_EACH and MIN_CHARS <= len(product_description) <= MAX_TEXT_EACH:

        full = scrub(product_title, product_description)
        if len(full) >= MIN_CHARS:
            return Item(
                title=product_title,
                category=ground_truth_category,
                full=full,
            )




