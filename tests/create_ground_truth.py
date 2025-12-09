#!/usr/bin/env python3
"""Interactive tool to create ground truth matcher dataset."""
import json
import pandas as pd
import yaml
from pathlib import Path

# Load data
def load_data():
    # Load Target products
    target = []
    with open('data/target_final.json') as f:
        for line in f:
            target.append(json.loads(line.strip()))

    # Load competitors (exclude Target source)
    competitor = pd.read_csv('data/harmonized_competitor_products.csv')
    competitor = competitor[competitor['source'].str.upper() != 'TARGET'].copy()
    competitor = competitor[competitor['source'].notna()].copy()

    return target, competitor.to_dict('records')

def extract_screen_size(target_product):
    """Extract screen size from Target product specs."""
    specs = target_product.get('product_specifications', [])
    for spec in specs:
        if spec.get('specification_name') == 'Screen Size':
            val = spec.get('specification_value', '')
            if 'Inches' in val:
                return float(val.split()[0])
    return None

def search_competitors(target_prod, competitors, brand=None, screen_size=None, tolerance=2):
    """Find candidate matches in competitor data."""
    candidates = []

    for comp in competitors:
        # Filter by brand if specified
        if brand and comp.get('brand', '').lower() != brand.lower():
            continue

        # Filter by screen size if specified
        if screen_size:
            comp_size = comp.get('screen_size_inches')
            if comp_size and abs(comp_size - screen_size) > tolerance:
                continue

        candidates.append(comp)

    return candidates

def display_product(prod, is_target=True):
    """Display product info for manual review."""
    if is_target:
        print(f"  Brand: {prod.get('product_brand')}")
        print(f"  Title: {prod.get('title')}")
        print(f"  Price: ${prod.get('final_price')}")
        print(f"  Screen: {extract_screen_size(prod)} inches")
    else:
        print(f"  Source: {prod.get('source')}")
        print(f"  Brand: {prod.get('brand')}")
        print(f"  Title: {prod.get('title')}")
        print(f"  SKU: {prod.get('sku')}")
        print(f"  Price: ${prod.get('final_price')}")
        print(f"  Screen: {prod.get('screen_size_inches')} inches")
        print(f"  Platform: {prod.get('smart_platform')}")

def main():
    target, competitors = load_data()
    print(f"Loaded {len(target)} Target products, {len(competitors)} competitors\n")

    # Sample strategy: Get diverse products
    # For now, just take first 20 as a starting point
    sample_indices = list(range(min(20, len(target))))

    ground_truth = []

    for idx in sample_indices:
        tprod = target[idx]
        screen_size = extract_screen_size(tprod)
        brand = tprod.get('product_brand')

        print(f"\n{'='*80}")
        print(f"TARGET PRODUCT #{idx}")
        print('='*80)
        display_product(tprod, is_target=True)

        # Search for candidates
        candidates = search_competitors(tprod, competitors, brand=brand, screen_size=screen_size, tolerance=3)

        if not candidates:
            print("\n  ❌ NO CANDIDATES FOUND")
            continue

        print(f"\n  Found {len(candidates)} candidates (same brand, ±3\" screen size)")
        print("\n  Top 5 candidates:")
        for i, cand in enumerate(candidates[:5], 1):
            print(f"\n  CANDIDATE #{i}:")
            display_product(cand, is_target=False)

        # Manual selection
        print("\n" + "-"*80)
        choice = input("  Enter candidate number for TRUE match (1-5, or 's' to skip, 'm' for more): ").strip()

        if choice.lower() == 's':
            print("  ⏭️  Skipped")
            continue
        elif choice.lower() == 'm':
            # Show more candidates
            for i, cand in enumerate(candidates[5:15], 6):
                print(f"\n  CANDIDATE #{i}:")
                display_product(cand, is_target=False)
            choice = input("\n  Enter candidate number (or 's' to skip): ").strip()
            if choice.lower() == 's':
                continue

        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(candidates):
                print("  ❌ Invalid choice")
                continue

            match = candidates[choice_idx]
            rationale = input("  Rationale (why is this a match?): ").strip()

            # Add to ground truth
            gt_entry = {
                'target_index': idx,
                'target_title': tprod.get('title'),
                'target_brand': brand,
                'target_price': tprod.get('final_price'),
                'target_screen_size': screen_size,
                'expected_competitor_sku': match.get('sku'),
                'expected_source': match.get('source'),
                'expected_title': match.get('title'),
                'expected_brand': match.get('brand'),
                'expected_price': match.get('final_price'),
                'expected_screen_size': match.get('screen_size_inches'),
                'match_rationale': rationale or 'Manual verification'
            }
            ground_truth.append(gt_entry)
            print(f"  ✅ Added to ground truth ({len(ground_truth)}/20)")

        except ValueError:
            print("  ❌ Invalid input")
            continue

        if len(ground_truth) >= 20:
            print("\n🎯 Collected 20 ground truth cases!")
            break

    # Save to YAML
    output_path = Path('tests/matcher_ground_truth.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(ground_truth, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Saved ground truth to {output_path}")
    print(f"   Total cases: {len(ground_truth)}")

if __name__ == '__main__':
    main()
