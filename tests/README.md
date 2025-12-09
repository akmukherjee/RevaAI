# Matcher Ground Truth Dataset

This directory contains ground truth data for testing and evaluating the SKU matcher component of Reva AI.

## Files

- **`matcher_ground_truth.yaml`** - Positive examples (14 cases): True matches between Target products and competitor products
- **`matcher_negative_examples.yaml`** - Negative examples (13 cases): Incorrect/poor matches (different product tiers, technologies, or platforms)
- **`create_ground_truth.py`** - Script used to generate the ground truth dataset

## Dataset Overview

### Positive Examples (14 cases)
These represent TRUE matches where Target and competitor products are the same model:

- **7 high-quality exact matches**: Same model number, exact screen size, >70% title similarity
- **7 questionable matches**: Different display tech (QLED→LED), platform mismatches (Google TV→Fire TV), or renewed products

**Distribution:**
- Sources: Amazon (10), Best Buy (4)
- Brands: VIZIO (8), TCL (4), Samsung (2)
- Screen sizes: 40"-85"
- Title similarity: 56-80%

### Negative Examples (13 cases)
These represent INCORRECT matches that should be avoided:

- Different product tiers (budget S5-Series matched to premium QM8K)
- Huge price differences (>100%)
- Low title similarity (<60%)

## Creation Methodology

### Data Sources
- **Internal Catalog:** Target products (`data/target_final.json`, 260 products)
- **Competitor Data:** Harmonized competitor products (`data/harmonized_competitor_products.csv`, 1,391 products from Amazon, Walmart, Best Buy)

### Selection Criteria

**Positive matches must meet:**
1. **Same brand** (case-insensitive)
2. **Similar screen size** (±1 inch tolerance)
3. **High title similarity** (>65% using SequenceMatcher)
4. **Reasonable price difference** (<100%)
5. **Same product tier** (manual verification)

**Why only 14 cases?**
The dataset has limited overlap between Target's catalog and competitor products. Most Target products either:
- Don't have matching models in competitor data (different inventory)
- Have very low title similarity (<60%, indicating different models)
- Have huge price differences (>100%, indicating different tiers)

After exhaustive search across all 260 Target products, only ~14 high-quality matches exist in the current dataset.

## Usage

### Reproducing the Dataset

```bash
# Run the ground truth generation script
python tests/create_ground_truth.py

# Note: This is an interactive script that requires manual verification
# The automated version with criteria is embedded in the script
```

### Testing the Matcher

```python
import yaml

# Load ground truth
with open('tests/matcher_ground_truth.yaml') as f:
    ground_truth = yaml.safe_load(f)

# Load negative examples
with open('tests/matcher_negative_examples.yaml') as f:
    negative_examples = yaml.safe_load(f)

# Test your matcher
for case in ground_truth:
    result = your_matcher(
        target_index=case['target_index'],
        competitor_data=load_competitor_data()
    )

    # Verify it matches the expected competitor
    assert result['sku'] == case['expected_competitor_sku']
```

## Schema

### Ground Truth Entry Format

```yaml
- target_index: 6                    # Index in target_final.json
  target_title: "VIZIO 55\" Class..."  # Target product title
  target_brand: "VIZIO"              # Target brand
  target_price: 237.99               # Target price (USD)
  target_screen_size: 55.0           # Target screen size (inches)

  expected_competitor_sku: "nan"     # Expected match SKU (may be NaN for Amazon)
  expected_source: "Amazon"          # Expected source (Amazon/Walmart/Best Buy)
  expected_title: "Vizio V4K55M..." # Expected competitor title
  expected_brand: "VIZIO"            # Expected competitor brand
  expected_price: 275.0              # Expected competitor price
  expected_screen_size: 55.0         # Expected screen size

  match_rationale: "Same brand, exact screen size, 70% title similarity"
  title_similarity: "0.70"           # Similarity score (0-1)
```

## Limitations

1. **Dataset size**: Only 14 positive cases due to limited product overlap
2. **Amazon SKUs**: Amazon products have `nan` for SKU (they use ASINs instead)
3. **Price volatility**: Prices may change over time; use price differences as relative indicators
4. **Model year variations**: Some matches may be from different years (2023 vs 2024 models)
5. **Platform mismatches**: Some "questionable" matches have different smart platforms (Google TV vs Fire TV) despite being the same series

## Future Improvements

To expand the ground truth dataset:

1. **Acquire more overlapping data**: Scrape additional retailers or products
2. **Manual curation**: Manually verify and add edge cases
3. **Synthetic examples**: Create synthetic test cases for specific scenarios
4. **Cross-retailer matching**: Use UPC/model numbers to find exact matches across all retailers

## References

- [Project Proposal](https://docs.google.com/document/d/1WUaQGXIU26Mv5wnYcBkqG5fiUQqPgHRtA9YWkz58iZo/edit?tab=t.0)
- [Data Harmonization Script](../notebooks/data_processing/harmonize.py)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
