#!/usr/bin/env python3
"""Embedding experiment for SKU matching optimization."""
import json
import os
import hashlib
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Experiment configuration
NUM_TARGET_PRODUCTS = 2  # Number of target products to test (change this to scale experiment)
BATCH_SIZE = 100  # Number of texts to embed per API call
CACHE_DIR = Path("data/embeddings")  # Directory for embedding cache

# Column combinations to test (all 5)
COMBINATIONS = {
    "baseline": ["title", "brand", "description"],
    "with_features": ["title", "brand", "features"],
    "with_model": ["title", "model_number", "description"],
    "structured_specs": ["title", "brand", "screen_size_inches", "resolution", "display_technology"],
    "full_context": ["title", "brand", "description", "features"],
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_text(product, columns):
    """Build embedding text from product columns."""
    parts = []
    for col in columns:
        # Handle brand field name variations (Target uses 'product_brand', competitors use 'brand')
        if col == "brand":
            val = product.get("brand") or product.get("product_brand")
        else:
            val = product.get(col)

        if val and str(val) not in ["None", "nan", ""]:
            parts.append(str(val))
    return " | ".join(parts)


def get_embedding(text):
    """Get embedding from OpenAI (single text - use for individual queries only)."""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def batch_get_embeddings(texts):
    """Get embeddings for multiple texts in batches."""
    all_embeddings = []

    # Process in batches of BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

        response = client.embeddings.create(input=batch, model="text-embedding-3-small")
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_cache_path(dataset_name, combo_name):
    """Generate cache file path for embeddings (hash-based)."""
    cache_key = f"{dataset_name}_{combo_name}"
    return CACHE_DIR / f"{cache_key}.pkl"


def migrate_npy_to_pkl_cache(old_cache_path, new_cache_path, products, columns):
    """Migrate position-based .npy cache to content-based .pkl cache."""
    if not old_cache_path.exists() or new_cache_path.exists():
        return  # Nothing to migrate or already migrated

    print(f"  Migrating old cache from {old_cache_path.name} to {new_cache_path.name}...")
    old_embeddings = np.load(old_cache_path)

    # Build hash-based cache from position-based cache
    cache_dict = {}
    for i, prod in enumerate(products):
        text = build_text(prod, columns)
        if text and i < len(old_embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_dict[text_hash] = old_embeddings[i]

    # Save new cache
    with open(new_cache_path, 'wb') as f:
        pickle.dump(cache_dict, f)

    print(f"  Migrated {len(cache_dict)} embeddings to hash-based cache")


def get_or_create_embeddings(products, columns, combo_name, dataset_name):
    """Get embeddings from cache or generate if not cached (content-based caching)."""
    cache_path = get_cache_path(dataset_name, combo_name)
    old_cache_path = CACHE_DIR / f"{dataset_name}_{combo_name}.npy"

    # Try to migrate old cache if it exists
    migrate_npy_to_pkl_cache(old_cache_path, cache_path, products, columns)

    # Load existing cache or create new one
    if cache_path.exists():
        print(f"  Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_dict = pickle.load(f)
    else:
        cache_dict = {}

    # Build embeddings array, using cache where possible
    embeddings_array = np.zeros((len(products), 1536))
    texts_to_embed = []
    indices_to_embed = []
    cache_hits = 0

    for i, prod in enumerate(products):
        text = build_text(prod, columns)
        if not text:
            continue  # Leave as zero vector

        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in cache_dict:
            # Cache hit - reuse existing embedding
            embeddings_array[i] = cache_dict[text_hash]
            cache_hits += 1
        else:
            # Cache miss - need to embed
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # Embed new texts if needed
    if texts_to_embed:
        print(f"  Cache hits: {cache_hits}/{len(products)}, generating {len(texts_to_embed)} new embeddings...")
        new_embeddings = batch_get_embeddings(texts_to_embed)

        # Update array and cache
        for idx, embedding in zip(indices_to_embed, new_embeddings):
            embeddings_array[idx] = embedding
            text = build_text(products[idx], columns)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_dict[text_hash] = embedding

        # Save updated cache
        print(f"  Saving updated cache to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_dict, f)
    else:
        print(f"  All {cache_hits} embeddings loaded from cache (0 API calls needed)")

    return embeddings_array


def load_data():
    """Load target and competitor data."""
    target = []
    with open("data/target_final.json") as f:
        for line in f:
            target.append(json.loads(line.strip()))
    print("\n========== TARGET BRAND SANITY CHECK ==========")
    for i, t in enumerate(target[:5]):
        print(i, "TITLE:", t.get("product_name") or t.get("title"))
        print(i, "BRAND:", t.get("product_brand"))

    # Load competitor data and filter out Target products
    competitor = pd.read_csv("data/harmonized_competitor_products.csv")

    # ✅ CRITICAL: Exclude Target products from competitor set
    original_count = len(competitor)
    competitor = competitor[competitor['source'].str.upper() != 'TARGET'].copy()
    competitor = competitor[competitor['source'].notna()].copy()  # Also remove empty source
    filtered_count = len(competitor)

    print(f"\n⚠️  FILTERED OUT {original_count - filtered_count} Target/unknown products from competitors")
    print(f"✅ Using {filtered_count} competitor products (Amazon, Walmart, Best Buy only)")

    competitor_records = competitor.to_dict("records")

    # ✅✅✅ DEBUG 3: GLOBAL SHARED BRAND CHECK (OPTIONAL BUT VERY POWERFUL)
    target_brands = {t.get("product_brand") for t in target if t.get("product_brand")}
    competitor_brands = {
        c.get("brand") or c.get("product_brand") or c.get("brand_name")
        for c in competitor_records
        if (c.get("brand") or c.get("product_brand") or c.get("brand_name"))
    }

    print("\n========== SHARED BRAND UNIVERSE ==========")
    print(target_brands & competitor_brands)
    return target, competitor_records


def brand_score(query, matches):
    """% of matches with same brand."""
    # Handle both 'brand' and 'product_brand' field names
    query_brand = str(query.get("brand") or query.get("product_brand", "")).lower()
    if not query_brand or query_brand == "nan":
        return 0.0

    same = 0
    for m in matches:
        match_brand = str(m.get("brand") or m.get("product_brand", "")).lower()
        if match_brand == query_brand:
            same += 1

    return same / len(matches)


def price_score(query, matches):
    """% of matches within ±30% price."""
    query_price = query.get("final_price") or query.get("initial_price")
    if not query_price:
        return 0.0
    in_range = 0
    for m in matches:
        m_price = m.get("final_price") or m.get("initial_price")
        if m_price and abs(m_price - query_price) / query_price <= 0.3:
            in_range += 1
    return in_range / len(matches)


def run_experiment(combo_name, columns, target, competitor):
    """Run experiment for one column combination."""
    print(f"\n=== {combo_name}: {columns} ===")

    # Embed competitor products (with caching)
    print("Embedding competitor products...")
    comp_embeds = get_or_create_embeddings(competitor, columns, combo_name, "competitor")

    # Test on target products
    print(f"Testing on first {NUM_TARGET_PRODUCTS} target products...")
    brand_scores = []
    price_scores = []
    all_matches = []  # Store all matches to return

    for i, query_prod in enumerate(target[:NUM_TARGET_PRODUCTS]):
        query_text = build_text(query_prod, columns)
        if not query_text:
            continue

        query_embed = np.array([get_embedding(query_text)])
        sims = cosine_similarity(query_embed, comp_embeds)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]

        matches = [competitor[idx] for idx in top5_idx]
        similarities = [sims[idx] for idx in top5_idx]

        # Store matches with similarity scores
        all_matches.append({
            'query': query_prod,
            'matches': matches,
            'similarities': similarities
        })

        # ✅ Keep scoring
        brand_scores.append(brand_score(query_prod, matches))
        price_scores.append(price_score(query_prod, matches))

    avg_brand = np.mean(brand_scores) if brand_scores else 0
    avg_price = np.mean(price_scores) if price_scores else 0
    overall = (avg_brand + avg_price) / 2

    print(f"Brand consistency: {avg_brand:.2%}")
    print(f"Price proximity: {avg_price:.2%}")
    print(f"Overall score: {overall:.2%}")

    return {
        "combination": combo_name,
        "columns": ",".join(columns),
        "brand_score": avg_brand,
        "price_score": avg_price,
        "overall_score": overall,
        "all_matches": all_matches,  # Include match details
    }


def main():
    """Run experiment."""
    target, competitor = load_data()
    print(f"Loaded {len(target)} target products, {len(competitor)} competitor products")

    results = []
    for name, cols in COMBINATIONS.items():
        result = run_experiment(name, cols, target, competitor)
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("data/experiment_results.csv", index=False)
    print(f"\n✓ Results saved to data/experiment_results.csv")
    print(df)

    # ✅ NEW: Compare matches across all combinations for first query product
    print("\n" + "="*80)
    print("MATCH COMPARISON ACROSS ALL 5 COMBINATIONS (First Target Product)")
    print("="*80)

    if results and results[0]['all_matches']:
        query_prod = results[0]['all_matches'][0]['query']
        print(f"\nQUERY: {query_prod.get('product_name') or query_prod.get('title')}")
        print(f"BRAND: {query_prod.get('product_brand')}")
        print(f"PRICE: ${query_prod.get('final_price') or query_prod.get('initial_price')}")

        for result in results:
            combo_name = result['combination']
            matches = result['all_matches'][0]['matches']
            similarities = result['all_matches'][0]['similarities']

            print(f"\n--- {combo_name.upper()} ---")
            for rank, (match, sim) in enumerate(zip(matches[:5], similarities[:5]), 1):
                title = match.get('title') or match.get('product_name', 'N/A')
                brand = match.get('brand') or match.get('product_brand', 'N/A')
                source = match.get('source', 'Unknown')
                price = match.get('final_price') or match.get('initial_price', 'N/A')
                print(f"  {rank}. [{source}] {brand} - {title[:60]}... (sim: {sim:.4f}, ${price})")


if __name__ == "__main__":
    main()
