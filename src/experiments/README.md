# SKU Matching Experiments

This directory contains experiments for optimizing the SKU matching component of Reva AI.

## Overview

The matcher needs to find corresponding products across different retailers (Target → Amazon/Walmart/Best Buy). We use embeddings to measure semantic similarity between products, but which product fields should we embed?

This experiment tests 5 different embedding strategies to determine the optimal approach.

## Files

- **`run_embedding_experiment.py`** - Main experiment script (tests 5 embedding combinations)

## Experiment Design

### Goal
Find the best product field combination for generating embeddings that accurately match products across retailers.

### Approach
1. **Test 5 combinations** of product fields (title, brand, features, specs, etc.)
2. **Embed all competitor products** (Amazon, Walmart, Best Buy) using each combination
3. **For each Target product**, find top-5 most similar competitors using cosine similarity
4. **Score results** based on:
   - **Brand consistency**: % of matches with same brand
   - **Price proximity**: % of matches within ±30% price range
5. **Compare** all combinations to identify best performer

### Embedding Combinations

| Combination | Fields Used | Strategy |
|-------------|-------------|----------|
| **baseline** | title, brand, description | Basic product info |
| **with_features** | title, brand, features | Emphasize product features |
| **with_model** | title, model_number, description | Emphasize model matching |
| **structured_specs** | title, brand, screen_size, resolution, display_tech | Structured specs only |
| **full_context** | title, brand, description, features | Maximum context |

### Scoring Metrics

- **Brand Score**: Percentage of top-5 matches that are the same brand as query
- **Price Score**: Percentage of top-5 matches within ±30% price range
- **Overall Score**: Average of brand and price scores

Higher scores indicate better matching quality.

## Usage

### Prerequisites

```bash
# Ensure environment variables are set
export OPENAI_API_KEY="your-key-here"

# Ensure data files exist
ls data/target_final.json
ls data/harmonized_competitor_products.csv
```

### Run Experiment

```bash
# Run with default settings (2 target products for quick test)
python src/experiments/run_embedding_experiment.py

# Results saved to: data/experiment_results.csv
```

### Configuration

Edit the script to adjust experiment parameters:

```python
# Number of target products to test
NUM_TARGET_PRODUCTS = 2  # Change to 260 for full evaluation

# Batch size for API calls
BATCH_SIZE = 100  # Max recommended: 100

# Cache directory
CACHE_DIR = Path("data/embeddings")  # Embeddings cached here
```

### Output

The script produces:

1. **CSV Results**: `data/experiment_results.csv` with scores for each combination
2. **Console Output**: Detailed match comparison showing top-5 matches for first query
3. **Embedding Cache**: Saved to `data/embeddings/` (prevents redundant API calls)

#### Example Output

```
=== baseline: ['title', 'brand', 'description'] ===
Embedding competitor products...
  All 1391 embeddings loaded from cache (0 API calls needed)
Testing on first 2 target products...
Brand consistency: 80.00%
Price proximity: 60.00%
Overall score: 70.00%

=== with_features: ['title', 'brand', 'features'] ===
...
```

```csv
combination,columns,brand_score,price_score,overall_score
baseline,"title,brand,description",0.80,0.60,0.70
with_features,"title,brand,features",0.90,0.70,0.80
...
```

## Caching

The experiment uses **content-based caching** to avoid redundant OpenAI API calls:

- **Cache key**: MD5 hash of embedding text
- **Cache location**: `data/embeddings/`
- **Format**: Pickle files (`.pkl`)
- **Auto-migration**: Old `.npy` caches automatically migrated to new format

Benefits:
- ✅ Avoids re-embedding same products
- ✅ Persists across runs
- ✅ Reduces API costs
- ✅ Faster iteration

**Cache invalidation**: Delete cache files to force re-embedding (e.g., after changing column combinations)

```bash
rm -rf data/embeddings/
```

## Interpreting Results

### What to Look For

1. **High brand consistency** (>80%) - Indicates embeddings capture brand identity well
2. **High price proximity** (>60%) - Indicates embeddings match similar product tiers
3. **High overall score** (>70%) - Good general matching quality

### Common Patterns

- **baseline** performs well but may miss model-specific details
- **with_model** excels when model numbers are present and consistent
- **structured_specs** works best for standardized categories (TVs, appliances)
- **full_context** provides maximum information but may include noise

### Decision Criteria

Choose the combination that:
1. Achieves highest overall score
2. Balances brand consistency and price proximity
3. Generalizes well across product categories

## Cost Estimation

**OpenAI Embedding Costs** (text-embedding-3-small):
- **Price**: $0.02 per 1M tokens (~$0.00002 per embedding)
- **Competitor dataset**: ~1,391 products × 5 combinations = 6,955 embeddings = ~$0.14
- **Query products**: 260 × 5 combinations = 1,300 embeddings = ~$0.03
- **Total (full run)**: ~$0.17

**With caching**: Only first run incurs costs. Subsequent runs are free!

## Limitations

1. **Small test set**: Default uses only 2 Target products for quick testing
2. **Static embeddings**: Doesn't account for seasonal variations or new products
3. **Binary scoring**: Doesn't capture nuanced match quality (exact vs close match)
4. **No model number matching**: Some products may have identical model numbers but different UPCs

## Next Steps

1. **Run full evaluation**: Set `NUM_TARGET_PRODUCTS = 260` for comprehensive results
2. **Analyze edge cases**: Review products where all combinations fail
3. **Hybrid approach**: Combine embeddings with rule-based filters (brand, price range)
4. **Fine-tuning**: Consider fine-tuning embeddings on labeled match pairs (future work)

## References

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Text Embedding 3 Models](https://openai.com/blog/new-embedding-models-and-api-updates)
- [Cosine Similarity Explanation](https://en.wikipedia.org/wiki/Cosine_similarity)
