"""
Data Harmonization Script for Competitor Product Data

This script processes raw scraped product data from Amazon, Walmart, and Best Buy,
normalizing it into a unified schema suitable for pricing analysis and SKU matching.

Usage:
    python harmonize.py --input ~/data/ --pattern "*.json" --out-prefix harmonized --tv-only true

Output:
    - harmonized.csv: CSV format for spreadsheet viewing
    - harmonized.parquet: Parquet format for data processing
    - harmonized_data_dictionary.yaml: Column documentation
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# ------------------------
# Configuration
# ------------------------

# Maps source-specific specification field names to standardized column names
# This handles the fact that Amazon calls it "Display Type" while Walmart uses "Display Technology"
SPEC_KEYS_MAP = {
    # Display / Panel
    "Display Technology": "display_technology",
    "Display Type": "display_technology",
    "Display": "display_technology",  # Walmart uses just "Display"
    "LED Panel Type": "panel_type",
    "Backlight Type": "backlight_type",
    "High Dynamic Range (HDR)": "hdr",
    "High Dynamic Range Format": "hdr_formats",
    "Resolution": "resolution",
    "Refresh Rate": "refresh_rate",
    "Aspect Ratio": "aspect_ratio",
    "Standing screen display size": "screen_size",
    "Screen Size": "screen_size",
    "Screen Size Class": "screen_size_class",

    # Connectivity
    "Connectivity Technology": "connectivity",
    "Number of HDMI Inputs (Total)": "hdmi_inputs",
    "Number of HDMI Inputs": "hdmi_inputs",
    "HDMI number": "hdmi_inputs",
    "USB Ports": "usb_ports",
    "Number Of USB Port(s) (Total)": "usb_ports",
    "Wireless Connectivity": "wireless",
    "Wireless technology": "wireless",

    # Audio / Features
    "Built-In Speakers": "built_in_speakers",
    "Special Features": "special_features",
    "Motion Enhancement Technology": "motion_tech",

    # Physical / Dimensions
    "Product Dimensions": "product_dimensions",
    "Item Weight": "item_weight",
    "Product Width": "width",
    "Product Height": "height",
    "Product Depth": "depth",
    "Product Height Without Stand": "height_without_stand",
    "Product Depth Without Stand": "depth_without_stand",
    "Product Weight Without Stand": "weight_without_stand",

    # Mounting
    "VESA Wall Mount Standard": "vesa",
    "Vesa mounting pattern": "vesa",

    # Platform / OS
    "Smart Platform": "smart_platform",
    "Platform": "smart_platform",

    # Identity
    "Brand Name": "brand",
    "Brand": "brand",
    "Model Number": "model_number",
    "Item model number": "model_number",
    "ASIN": "asin",
    "UPC": "upc",
}

# Define the complete output schema with all columns in order
# This ensures consistent column ordering across all output files
COLUMNS = [
    # Source & metadata
    "source", "domain", "url", "title",

    # Product identity
    "brand", "model_number", "model_name", "series",
    "asin", "sku", "upc", "gtin",

    # Pricing
    "final_price", "initial_price", "currency", "discount", "deal_type",

    # Availability & logistics
    "availability_text", "is_available", "return_policy", "delivery_info",

    # Reviews & social proof
    "rating", "reviews_count", "top_review", "customers_say",

    # Categories & seller/store
    "categories", "badges", "seller_name", "seller_rating", "store_name", "store_location",

    # Display specifications
    "screen_size_inches", "screen_size_class", "resolution", "refresh_rate", "aspect_ratio",
    "display_technology", "panel_type", "backlight_type", "hdr", "hdr_formats",

    # Connectivity specifications
    "smart_platform", "voice_assistants", "wireless", "connectivity", "hdmi_inputs", "usb_ports",

    # Physical dimensions
    "product_dimensions", "item_weight", "height", "width", "depth",
    "height_without_stand", "depth_without_stand", "weight_without_stand", "vesa",

    # Content & media
    "features", "description", "images", "image_count",

    # Miscellaneous
    "timestamp", "bought_past_month", "amazon_prime", "badges_all", "raw_specs_json"
]

# Detailed descriptions for each column in the output schema
COLUMN_DESCRIPTIONS = {
    # Source & metadata
    "source": "Retailer source (Amazon, Walmart, Best Buy)",
    "domain": "Domain name or URL the product was scraped from",
    "url": "Direct product page URL",
    "title": "Product title/name as displayed on retailer site",

    # Product identity
    "brand": "Product brand/manufacturer",
    "model_number": "Manufacturer's model number",
    "model_name": "Marketing model name",
    "series": "Product series/line",
    "asin": "Amazon Standard Identification Number",
    "sku": "Stock Keeping Unit (retailer-specific identifier)",
    "upc": "Universal Product Code (barcode number)",
    "gtin": "Global Trade Item Number",

    # Pricing
    "final_price": "Current selling price (numeric)",
    "initial_price": "Original/list price before discounts (numeric)",
    "currency": "Price currency code (e.g., USD)",
    "discount": "Discount description or amount",
    "deal_type": "Type of deal/promotion",

    # Availability & logistics
    "availability_text": "Availability status description",
    "is_available": "Boolean availability flag",
    "return_policy": "Return policy description",
    "delivery_info": "Shipping/delivery information",

    # Reviews & social proof
    "rating": "Customer rating (out of 5)",
    "reviews_count": "Number of customer reviews",
    "top_review": "Excerpt from top customer review",
    "customers_say": "Summary of customer feedback",

    # Categories & seller/store
    "categories": "Product categories (semicolon-separated)",
    "badges": "Product badges/labels",
    "seller_name": "Third-party seller name (if applicable)",
    "seller_rating": "Seller rating score",
    "store_name": "Physical store name (if applicable)",
    "store_location": "Store location/address",

    # Display specifications
    "screen_size_inches": "Screen size in inches (numeric, e.g., 65.0)",
    "screen_size_class": "Screen size marketing class (e.g., '65-Inch')",
    "resolution": "Display resolution (e.g., '4K UHD', '1920x1080')",
    "refresh_rate": "Display refresh rate (e.g., '120Hz')",
    "aspect_ratio": "Screen aspect ratio (e.g., '16:9')",
    "display_technology": "Display technology type (LED, OLED, QLED, etc.)",
    "panel_type": "Panel type specification",
    "backlight_type": "Backlight technology",
    "hdr": "HDR (High Dynamic Range) support",
    "hdr_formats": "Supported HDR formats (e.g., HDR10, Dolby Vision)",

    # Connectivity specifications
    "smart_platform": "Smart TV platform (e.g., Roku, webOS, Fire TV)",
    "voice_assistants": "Supported voice assistants (e.g., Alexa, Google Assistant)",
    "wireless": "Wireless connectivity features (WiFi, Bluetooth)",
    "connectivity": "General connectivity options",
    "hdmi_inputs": "Number of HDMI input ports",
    "usb_ports": "Number of USB ports",

    # Physical dimensions
    "product_dimensions": "Overall dimensions with stand (W x H x D)",
    "item_weight": "Product weight with stand",
    "height": "Height with stand",
    "width": "Width (screen diagonal)",
    "depth": "Depth with stand",
    "height_without_stand": "Height without stand",
    "depth_without_stand": "Depth without stand",
    "weight_without_stand": "Weight without stand",
    "vesa": "VESA wall mount standard (e.g., 400x400)",

    # Content & media
    "features": "Product features list (semicolon-separated)",
    "description": "Full product description text",
    "images": "Product image URLs (semicolon-separated)",
    "image_count": "Number of product images available",

    # Miscellaneous
    "timestamp": "Data scrape timestamp",
    "bought_past_month": "Number of units bought in past month (Amazon-specific)",
    "amazon_prime": "Amazon Prime eligibility (Yes/No)",
    "badges_all": "All product badges and certifications",
    "raw_specs_json": "Original specifications as JSON string for traceability",
}

DATA_DICTIONARY = {
    "description": "Unified competitor product schema across Amazon, Walmart, and Best Buy.",
    "columns": COLUMN_DESCRIPTIONS,
    "notes": [
        "raw_specs_json stores the original spec key/value bag from each source for traceability.",
        "images stores a semicolon-separated list of image URLs.",
        "categories stores a semicolon-separated list of category labels.",
        "Semicolon-separated fields can be split on '; ' to recover list structure.",
    ],
}

# ------------------------
# Helpers
# ------------------------


def to_str(val):
    """Coerce any value to a string suitable for CSV/Parquet.
    - list -> 'a; b; c'
    - dict -> JSON string
    - None -> None
    - scalar -> str(scalar)
    """
    if val is None:
        return None
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v is not None)
    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return str(val)

def to_float(val: Any) -> Optional[float]:
    """Convert various price/numeric formats to float.

    Handles:
    - "$899.99" -> 899.99
    - "1,299.99" -> 1299.99
    - Already numeric values
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Strip non-numeric characters (except decimal point)
        s = re.sub(r"[^0-9.]+", "", val)
        return float(s) if s else None
    return None

def first_non_empty(*vals):
    """Return the first non-empty value from a list of alternatives.

    Used to handle different field names across sources.
    Example: first_non_empty(item.get("final_price"), item.get("offer_price"))
    """
    for v in vals:
        if v is not None and v != "" and v != []:
            return v
    return None

def unwrap_richtext_if_needed(obj: Any) -> List[Dict[str, Any]]:
    """
    Best Buy / Walmart exports sometimes use a dict like:
    { "<GUID>:content": { "type": "richtext", "value": { "text": "[ { ... }, { ... } ]" }}}
    This attempts to unwrap into a Python list.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        try:
            first_key = next(iter(obj))
            txt = obj[first_key]["value"]["text"]
            txt_clean = txt.replace("[This Output has been Truncated]", "")
            parsed = json.loads(txt_clean)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            # fallback: find any list inside values
            for v in obj.values():
                if isinstance(v, list):
                    return v
    return []

def collect_specs(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse all source-specific spec arrays into a single flat dict.
    Supports:
      - Amazon: product_details [{ type, value }]
      - Walmart: specifications [{ name, value }]
      - Best Buy: product_specifications [{ specification_name, specification_value }]
    """
    bag = {}
    for d in item.get("product_details", []) or []:
        k, v = d.get("type"), d.get("value")
        if k: bag[k] = v
    for s in item.get("specifications", []) or []:
        k, v = s.get("name"), s.get("value")
        if k: bag[k] = v
    for s in item.get("product_specifications", []) or []:
        k, v = s.get("specification_name"), s.get("specification_value")
        if k: bag[k] = v
    return bag

def normalize_specs(specs_flat: Dict[str, Any]) -> Dict[str, Any]:
    """Map source-specific spec keys to standardized column names.

    Uses SPEC_KEYS_MAP to convert retailer-specific field names into
    a unified schema (e.g., "Display Type" -> "display_technology").
    Performs case-insensitive matching to handle variations like
    "Screen Size" vs "Screen size" across retailers.
    """
    norm = {}
    for src_k, norm_k in SPEC_KEYS_MAP.items():
        # Find case-insensitive match in source data
        matched_key = None
        for k in specs_flat.keys():
            if k.lower() == src_k.lower():
                matched_key = k
                break

        if matched_key and specs_flat[matched_key] not in (None, ""):
            norm[norm_k] = specs_flat[matched_key]
    return norm

def infer_source(domain_or_url: Optional[str]) -> Optional[str]:
    """Detect retailer source from domain name or URL.

    Returns: "Amazon", "Walmart", "Best Buy", "Target", or None
    """
    s = (domain_or_url or "").lower()
    if "amazon" in s:
        return "Amazon"
    if "walmart" in s:
        return "Walmart"
    if "bestbuy" in s:
        return "Best Buy"
    if "target" in s:
        return "Target"
    return None

def extract_screen_inches(val: Optional[str]) -> Optional[float]:
    """Extract numeric screen size from strings like '65"', '65 inch', '65-Inch'.

    Returns: Float value (e.g., 65.0) or None
    """
    if not val:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    return float(m.group(1)) if m else None

def join_list(vals: Any) -> Optional[str]:
    """Convert list to semicolon-separated string for CSV/Parquet compatibility."""
    if isinstance(vals, list):
        return "; ".join(str(v) for v in vals if v is not None)
    return vals

def is_tv_item(item: Dict[str, Any], specs_flat: Dict[str, Any]) -> bool:
    """Check if item is actually a TV based on specs, not just keywords.

    Filters out false positives like TV stands, ATV/UTV products, and accessories
    by requiring both TV indicators AND actual TV specifications.
    """
    cats = item.get("categories")
    cats_str = (join_list(cats) or "").lower()
    title = (item.get("title") or item.get("product_name") or "").lower()

    # Check for TV keywords in title/categories
    tv_keywords = any(kw in cats_str or kw in title
                     for kw in ["television", "smart tv", "qled", "oled"])

    # Must have actual display technology specs (TVs have this, furniture/toys don't)
    display_tech = (
        specs_flat.get("Display Type") or
        specs_flat.get("Display Technology") or
        specs_flat.get("Display") or  # Walmart uses just "Display"
        ""
    ).lower()
    has_display = any(tech in display_tech
                     for tech in ["led", "oled", "qled", "lcd", "plasma"])

    # Must have screen size specification (TVs have this, games/accessories don't)
    has_screen_size = any(key in specs_flat and specs_flat[key]
                         for key in ["Screen Size", "Screen Size Class",
                                    "Standing screen display size", "screen size", "Screen size"])

    # Item must have TV indicators AND actual TV specs (display tech OR screen size)
    # This filters out TV stands, ATV products, and other false positives
    return (tv_keywords or "tv" in cats_str) and (has_display or has_screen_size)

# ------------------------
# Main pipeline
# ------------------------

def harmonize_records(records: List[Dict[str, Any]], tv_only: bool = True) -> List[Dict[str, Any]]:
    """Transform raw scraped records into standardized schema.

    This is the core harmonization function that:
    1. Extracts specifications from various source formats
    2. Normalizes field names to a unified schema
    3. Optionally filters to TV products only
    4. Constructs complete output records with all required columns

    Args:
        records: List of raw product dictionaries from scrapers
        tv_only: If True, filter to keep only TV products

    Returns:
        List of harmonized product dictionaries matching COLUMNS schema
    """
    rows = []
    for item in records:
        # Step 1: Extract all specifications into a flat dict
        specs_flat = collect_specs(item)
        # Step 2: Map to standardized field names
        normalized_specs = normalize_specs(specs_flat)

        # Optional TV-only filtering
        if tv_only and not is_tv_item(item, specs_flat):
            continue

        screen_size_str = first_non_empty(normalized_specs.get("screen_size"), normalized_specs.get("screen_size_class"))
        screen_size_inches = extract_screen_inches(screen_size_str)

        # Voice assistants: various locations across sources
        voice_assistants = first_non_empty(item.get("voice_assistants"), specs_flat.get("Virtual asst."))
        if not voice_assistants:
            va = specs_flat.get("Voice Assistant Built-in")
            ww = specs_flat.get("Works With")
            voice_assistants = ", ".join([x for x in [va, ww] if x]) or None

        # Price
        final_price = to_float(first_non_empty(item.get("final_price"), item.get("offer_price")))
        initial_price = to_float(item.get("initial_price"))
        currency = item.get("currency")
        discount = item.get("discount")
        deal_type = None
        if isinstance(item.get("prices_breakdown"), dict):
            deal_type = item["prices_breakdown"].get("deal_type")

        # Reviews
        rating = item.get("rating")
        reviews_count = first_non_empty(item.get("reviews_count"), item.get("review_count"))
        top_review = to_str(item.get("top_review"))
        customers_say = first_non_empty(item.get("customer_says"), item.get("customers_say"), item.get("customers\\_say"))
        if isinstance(customers_say, dict) and customers_say.get("text"):
            customers_say = customers_say["text"]
        customers_say = to_str(customers_say)

        # Identity / source
        source_domain = first_non_empty(item.get("domain"), item.get("origin_url"))
        source = infer_source(source_domain or item.get("url"))
        url = item.get("url")

        # Categories
        categories = to_str(join_list(item.get("categories")))

        # Images & features
        images_list = item.get("images") or item.get("image_urls")
        image_count = len(images_list) if isinstance(images_list, list) else None
        images = to_str(join_list(images_list) or item.get("image"))
        features = to_str(join_list(item.get("features")))

        # Seller & store
        seller_name = item.get("seller_name")
        seller_rating = item.get("buybox_seller_rating")
        store_name = item.get("store_name")
        store_location = item.get("store_location")

        # Availability / returns / delivery
        availability_text = first_non_empty(item.get("availability"), item.get("availability_text"))
        # normalize to string
        availability_text = to_str(availability_text)
        is_available = item.get("is_available")
        return_policy = to_str(item.get("return_policy"))
        delivery_info = to_str(join_list(item.get("delivery")))

        # Misc
        badges = to_str(first_non_empty(item.get("badge"), item.get("badges")))
        badges_all = to_str(item.get("all_badges"))
        timestamp = item.get("timestamp")
        bought_past_month = item.get("bought_past_month")
        amazon_prime = item.get("amazon_prime")

        row = {
            "source": source,
            "domain": source_domain,
            "url": url,
            "title": first_non_empty(item.get("title"), item.get("product_name")),

            # identity
            "brand": first_non_empty(item.get("brand"), normalized_specs.get("brand")),
            "model_number": normalized_specs.get("model_number"),
            "model_name": specs_flat.get("Model name"),
            "series": specs_flat.get("Series"),
            "asin": normalized_specs.get("asin"),
            "sku": first_non_empty(item.get("sku"), item.get("us_item_id"), item.get("product_id")),
            "upc": first_non_empty(item.get("upc"), normalized_specs.get("upc")),
            "gtin": item.get("gtin"),

            # pricing
            "final_price": final_price,
            "initial_price": initial_price,
            "currency": currency,
            "discount": discount,
            "deal_type": deal_type,

            # availability
            "availability_text": availability_text,
            "is_available": is_available,
            "return_policy": return_policy,
            "delivery_info": delivery_info,

            # reviews
            "rating": rating,
            "reviews_count": reviews_count,
            "top_review": top_review,
            "customers_say": customers_say,

            # categories & seller/store
            "categories": categories,
            "badges": badges,
            "seller_name": seller_name,
            "seller_rating": seller_rating,
            "store_name": store_name,
            "store_location": store_location,

            # display
            "screen_size_inches": screen_size_inches,
            "screen_size_class": normalized_specs.get("screen_size_class"),
            "resolution": normalized_specs.get("resolution"),
            "refresh_rate": normalized_specs.get("refresh_rate"),
            "aspect_ratio": normalized_specs.get("aspect_ratio"),
            "display_technology": normalized_specs.get("display_technology"),
            "panel_type": normalized_specs.get("panel_type"),
            "backlight_type": normalized_specs.get("backlight_type"),
            "hdr": normalized_specs.get("hdr"),
            "hdr_formats": normalized_specs.get("hdr_formats"),

            # connectivity
            "smart_platform": normalized_specs.get("smart_platform"),
            "voice_assistants": voice_assistants,
            "wireless": normalized_specs.get("wireless"),
            "connectivity": normalized_specs.get("connectivity"),
            "hdmi_inputs": normalized_specs.get("hdmi_inputs"),
            "usb_ports": normalized_specs.get("usb_ports"),

            # physical
            "product_dimensions": first_non_empty(normalized_specs.get("product_dimensions"), item.get("product_dimensions")),
            "item_weight": first_non_empty(normalized_specs.get("item_weight"), item.get("item_weight")),
            "height": normalized_specs.get("height"),
            "width": normalized_specs.get("width"),
            "depth": normalized_specs.get("depth"),
            "height_without_stand": normalized_specs.get("height_without_stand"),
            "depth_without_stand": normalized_specs.get("depth_without_stand"),
            "weight_without_stand": normalized_specs.get("weight_without_stand"),
            "vesa": normalized_specs.get("vesa"),

            # content
            "features": features,
            "description": to_str(first_non_empty(item.get("description"), item.get("product_description"))),
            "images": images,
            "image_count": image_count,

            # misc
            "timestamp": timestamp,
            "bought_past_month": bought_past_month,
            "amazon_prime": amazon_prime,
            "badges_all": badges_all,
            "raw_specs_json": json.dumps(specs_flat, ensure_ascii=False),
        }

        # Ensure all columns exist
        for c in COLUMNS:
            row.setdefault(c, None)
        rows.append(row)
    return rows

def load_records_from_folder(input_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    """Load product records from JSON files in a directory.

    Supports multiple JSON formats:
    - Regular JSON array: [{"product": "..."}]
    - Richtext-wrapped (Best Buy/Walmart exports)
    - Newline-delimited JSON (NDJSON): one object per line

    Args:
        input_dir: Directory containing JSON files
        pattern: Glob pattern for file matching (e.g., "*.json")

    Returns:
        Combined list of all product records from all files

    Raises:
        ValueError: If a file cannot be parsed in any supported format
    """
    all_records: List[Dict[str, Any]] = []
    for path in input_dir.glob(pattern):
        print(f"Processing: {path.name}")
        with path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Try parsing as regular JSON array
        try:
            obj = json.loads(content)
            records = unwrap_richtext_if_needed(obj)
            if not records and isinstance(obj, list):
                records = obj
            if records:
                all_records.extend(records)
                print(f"  ✓ Loaded {len(records)} records")
                continue
        except json.JSONDecodeError:
            pass

        # Try parsing as newline-delimited JSON (NDJSON)
        lines = [json.loads(line) for line in content.splitlines() if line.strip() and line.strip().startswith("{")]
        if lines:
            all_records.extend(lines)
            print(f"  ✓ Loaded {len(lines)} records (NDJSON)")
            continue

        # If we get here, neither format worked - fail loudly
        raise ValueError(f"Could not parse {path.name} as JSON or NDJSON")

    return all_records

def save_outputs(df: pd.DataFrame, out_prefix: str):
    """Save harmonized data to CSV, Parquet, and data dictionary YAML.

    Handles data type conversions to ensure Parquet compatibility:
    - Forces identifier columns (UPC, SKU, ASIN) to string type to preserve leading zeros
    - Converts price/rating columns to numeric, coercing invalid values to NaN

    Args:
        df: DataFrame with harmonized product data
        out_prefix: File name prefix for output files

    Output files:
        - {out_prefix}.csv: Human-readable CSV format
        - {out_prefix}.parquet: Columnar format for efficient data processing
        - {out_prefix}_data_dictionary.yaml: Schema documentation
    """
    csv_path = Path(f"{out_prefix}.csv")
    parquet_path = Path(f"{out_prefix}.parquet")
    dd_path = Path(f"{out_prefix}_data_dictionary.yaml")

    # Force string type for identifier columns to preserve leading zeros
    # Example: UPC "810153055280" would become 810153055280 (int) without this
    string_columns = [
        "upc", "sku", "asin", "gtin", "model_number", "series",
        "discount", "url", "domain", "title", "brand", "model_name",
        "currency", "deal_type", "availability_text", "return_policy",
        "delivery_info", "top_review", "customers_say", "categories",
        "badges", "seller_name", "store_name", "store_location",
        "screen_size_class", "resolution", "refresh_rate", "aspect_ratio",
        "display_technology", "panel_type", "backlight_type", "hdr",
        "hdr_formats", "smart_platform", "voice_assistants", "wireless",
        "connectivity", "hdmi_inputs", "usb_ports", "product_dimensions",
        "item_weight", "height", "width", "depth", "height_without_stand",
        "depth_without_stand", "weight_without_stand", "vesa",
        "features", "description", "images", "timestamp",
        "bought_past_month", "badges_all", "raw_specs_json"
    ]

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', None).replace('None', None)

    # Convert price/rating columns to numeric, coercing errors to NaN
    numeric_columns = ["final_price", "initial_price", "rating", "reviews_count",
                      "screen_size_inches", "image_count"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Write output files
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(DATA_DICTIONARY, f, sort_keys=False, allow_unicode=True)

    print(f"Saved: {csv_path}")
    print(f"Saved: {parquet_path}")
    print(f"Saved: {dd_path}")

def main():
    """Main entry point for the harmonization pipeline.

    Command-line arguments:
        --input: Directory containing source JSON files
        --pattern: Glob pattern for file matching (default: "*.json")
        --out-prefix: Output file name prefix (default: "harmonized_competitor_products")
        --tv-only: Filter to TV products only (default: "true")

    Pipeline steps:
        1. Load all JSON files from input directory
        2. Harmonize records into unified schema
        3. Create pandas DataFrame with standardized columns
        4. Save to CSV, Parquet, and data dictionary YAML
    """
    parser = argparse.ArgumentParser(
        description="Harmonize competitor product data across Amazon/Walmart/Best Buy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JSON files in Downloads folder, TV products only
  python harmonize.py --input ~/Downloads/final_datasets/

  # Process all products (not just TVs)
  python harmonize.py --input ~/data/ --tv-only false

  # Custom output file name
  python harmonize.py --input ~/data/ --out-prefix my_products
        """
    )
    parser.add_argument("--input", type=str, default="./", help="Input folder containing JSON files.")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for files.")
    parser.add_argument("--out-prefix", type=str, default="harmonized_competitor_products", help="Output file prefix.")
    parser.add_argument("--tv-only", type=str, default="true", help="Filter to TVs only (true/false).")
    args = parser.parse_args()

    input_dir = Path(args.input)
    tv_only = str(args.tv_only).lower() in ["true", "1", "yes", "y"]

    # Step 1: Load JSON files
    records = load_records_from_folder(input_dir, args.pattern)
    print(f"Loaded {len(records)} raw records.")

    # Step 2: Harmonize into unified schema
    rows = harmonize_records(records, tv_only=tv_only)
    print(f"Keeping {len(rows)} harmonized rows{' (TV-only)' if tv_only else ''}.")

    # Step 3: Create DataFrame with enforced column order
    df = pd.DataFrame(rows, columns=COLUMNS)

    # Step 4: Clean up string fields (remove extra whitespace)
    if "smart_platform" in df.columns:
        df["smart_platform"] = df["smart_platform"].astype(str).str.strip()

    # Step 5: Save outputs
    save_outputs(df, args.out_prefix)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
