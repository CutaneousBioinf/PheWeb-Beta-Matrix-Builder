"""
Step 4: Analyze embedding similarity using cosine similarity
Configure dataset in config.py before running.

Why cosine similarity for PheWAS embeddings?
============================================
Just as word embeddings capture semantic relationships through co-occurrence patterns,
our variant-phenotype embeddings capture semantic relationships through genetic regulation:

- Word embeddings: word ↔ word co-occurrence (linguistic context)
- Our embeddings: variant ↔ phenotype association (genetic regulation)

Both represent semantic information in vector space, where cosine similarity measures
the angle/direction between vectors - indicating similarity regardless of magnitude.

This analysis uses ALL embedding dimensions (not just the first 2-3 shown in PCA plots),
capturing the full semantic information in the embeddings.

Three types of similarity:
1. Phenotype-Phenotype: Which diseases share genetic architecture?
2. Variant-Variant: Which variants have similar phenotypic effects?
3. Variant-Phenotype (cross): Which variants have similar roles to certain phenotypes?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

# Import configuration
from config import DATASET_NAME, EMBEDDINGS_DIR, SIMILARITY_DIR, setup_directories

# Memory efficiency settings
VARIANT_SIZE_THRESHOLD = 10000  # Use efficient method if more variants than this
TOP_K_SIMILAR = 100  # Number of top similar items to keep per variant (for large datasets)
CHUNK_SIZE = 1000  # Process this many items at a time

def load_embeddings():
    """Load variant and phenotype embeddings from CSV files"""
    variant_file = EMBEDDINGS_DIR / 'variant_embeddings.csv'
    phenotype_file = EMBEDDINGS_DIR / 'phenotype_embeddings.csv'

    if not variant_file.exists() or not phenotype_file.exists():
        print(f"Error: Embedding files not found in {EMBEDDINGS_DIR}")
        print("Run 3_create_embeddings.py first!")
        sys.exit(1)

    print(f"Loading embeddings from {EMBEDDINGS_DIR}")
    variant_df = pd.read_csv(variant_file, index_col=0)
    phenotype_df = pd.read_csv(phenotype_file, index_col=0)

    print(f"  Total variants loaded: {variant_df.shape[0]:,}")
    print(f"  Phenotypes loaded: {phenotype_df.shape[0]:,}")

    # Filter variants to only those with non-zero embeddings
    # (variants with no significant associations have zero embeddings)
    nonzero_mask = (variant_df != 0).any(axis=1)
    n_filtered = (~nonzero_mask).sum()

    if n_filtered > 0:
        print(f"  Filtering out {n_filtered:,} variants with zero embeddings (no significant associations)")
        variant_df = variant_df[nonzero_mask]

    print(f"  Final variants for analysis: {variant_df.shape[0]:,} x {variant_df.shape[1]} dimensions")
    print(f"  Phenotypes for analysis: {phenotype_df.shape[0]:,} x {phenotype_df.shape[1]} dimensions")

    return variant_df, phenotype_df

def compute_topk_similarity(embeddings_df, name, top_k=TOP_K_SIMILAR):
    """
    Memory-efficient computation of top-K most similar items for each item.
    Returns a sparse representation instead of full matrix.
    """
    print(f"Computing top-{top_k} similarities for {name}...")

    n_items = len(embeddings_df)
    embeddings = embeddings_df.values.astype(np.float32)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-10)

    # Store top-k results
    all_indices = []
    all_similarities = []
    all_rows = []

    # Adjust top_k if necessary
    actual_top_k = min(top_k, n_items - 1)

    # Process in chunks
    for start_idx in range(0, n_items, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, n_items)

        # Compute similarity for this chunk against all items
        chunk_embeddings = embeddings_normalized[start_idx:end_idx]
        similarities = chunk_embeddings @ embeddings_normalized.T

        # For each item in chunk, find top-k most similar
        for i in range(end_idx - start_idx):
            row_idx = start_idx + i
            sims = similarities[i]

            # Sort in descending order and get top-k (excluding self)
            sorted_indices = np.argsort(-sims)
            sorted_indices_no_self = sorted_indices[sorted_indices != row_idx][:actual_top_k]
            top_sims = sims[sorted_indices_no_self]

            # Store results
            all_rows.extend([row_idx] * len(sorted_indices_no_self))
            all_indices.extend(sorted_indices_no_self.tolist())
            all_similarities.extend(top_sims.tolist())

    # Create results dictionary for easy querying
    results = {
        'row_indices': np.array(all_rows),
        'col_indices': np.array(all_indices),
        'similarities': np.array(all_similarities),
        'index': embeddings_df.index,
        'n_items': n_items,
        'top_k': actual_top_k
    }

    print(f"  Stored {len(all_similarities):,} similarity values ({100 * len(all_similarities) / (n_items * n_items):.2f}% of full matrix)")
    return results

def compute_similarity_matrix(embeddings_df, name, use_efficient=False, top_k=TOP_K_SIMILAR):
    """Compute pairwise cosine similarity matrix"""
    n_items = len(embeddings_df)

    # Auto-select efficient mode for large datasets
    if n_items > VARIANT_SIZE_THRESHOLD:
        use_efficient = True
        print(f"Large dataset detected ({n_items:,} items) - using memory-efficient mode")

    if use_efficient:
        return compute_topk_similarity(embeddings_df, name, top_k)
    else:
        print(f"Computing full similarity matrix for {name}...")
        similarity_matrix = cosine_similarity(embeddings_df.values)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=embeddings_df.index,
            columns=embeddings_df.index
        )
        return similarity_df

def compute_cross_similarity(variant_df, phenotype_df, top_k=20):
    """
    Compute cross-similarity between variants and phenotypes.
    Since they're in the same latent space, we can compare them directly.
    """
    print(f"Computing variant-phenotype cross-similarity (top-{top_k} per variant)...")

    n_variants = len(variant_df)
    n_phenotypes = len(phenotype_df)

    # Normalize embeddings
    variant_emb = variant_df.values.astype(np.float32)
    phenotype_emb = phenotype_df.values.astype(np.float32)

    variant_norms = np.linalg.norm(variant_emb, axis=1, keepdims=True)
    phenotype_norms = np.linalg.norm(phenotype_emb, axis=1, keepdims=True)

    variant_normalized = variant_emb / (variant_norms + 1e-10)
    phenotype_normalized = phenotype_emb / (phenotype_norms + 1e-10)

    # Store results
    all_variant_indices = []
    all_phenotype_indices = []
    all_similarities = []

    # Process in chunks
    for start_idx in range(0, n_variants, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, n_variants)

        # Compute cross-similarity for this chunk
        chunk_variants = variant_normalized[start_idx:end_idx]
        cross_sims = chunk_variants @ phenotype_normalized.T

        # For each variant in chunk, find top-k most similar phenotypes
        for i in range(end_idx - start_idx):
            variant_idx = start_idx + i
            sims = cross_sims[i]

            # Get top-k phenotypes
            top_indices = np.argsort(-sims)[:top_k]
            top_sims = sims[top_indices]

            # Store results
            all_variant_indices.extend([variant_idx] * len(top_indices))
            all_phenotype_indices.extend(top_indices.tolist())
            all_similarities.extend(top_sims.tolist())

    results = {
        'variant_indices': np.array(all_variant_indices),
        'phenotype_indices': np.array(all_phenotype_indices),
        'similarities': np.array(all_similarities),
        'variant_index': variant_df.index,
        'phenotype_index': phenotype_df.index,
        'top_k': top_k
    }

    print(f"  Stored {len(all_similarities):,} cross-similarity values")
    return results

def save_similarity_data(similarity_data, output_path, data_type='within'):
    """Save similarity matrix to CSV"""
    if isinstance(similarity_data, dict):
        # Efficient mode - save as edge list
        if data_type == 'cross':
            # Cross-similarity format
            edge_list = pd.DataFrame({
                'variant': [similarity_data['variant_index'][i] for i in similarity_data['variant_indices']],
                'phenotype': [similarity_data['phenotype_index'][i] for i in similarity_data['phenotype_indices']],
                'similarity': similarity_data['similarities']
            })
        else:
            # Within-type similarity format
            index = similarity_data['index']
            edge_list = pd.DataFrame({
                'item1': [index[i] for i in similarity_data['row_indices']],
                'item2': [index[i] for i in similarity_data['col_indices']],
                'similarity': similarity_data['similarities']
            })

        edge_list.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name}")
    else:
        # Full mode - save as matrix
        similarity_data.to_csv(output_path)
        print(f"  Saved: {output_path.name}")

def analyze_top_pairs(similarity_data, top_k=20, data_type='within'):
    """Find and return the most similar pairs"""
    if data_type == 'cross':
        # Cross-similarity pairs
        variant_indices = similarity_data['variant_indices']
        phenotype_indices = similarity_data['phenotype_indices']
        sims = similarity_data['similarities']

        # Get top-k highest similarities
        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_indices:
            variant = similarity_data['variant_index'][variant_indices[idx]]
            phenotype = similarity_data['phenotype_index'][phenotype_indices[idx]]
            sim = sims[idx]
            results.append({'variant': variant, 'phenotype': phenotype, 'similarity': sim})

        return pd.DataFrame(results)

    elif isinstance(similarity_data, dict):
        # Within-type efficient mode
        index = similarity_data['index']
        row_indices = similarity_data['row_indices']
        col_indices = similarity_data['col_indices']
        sims = similarity_data['similarities']

        # Create unique pairs (avoid duplicates)
        pairs = []
        for i in range(len(row_indices)):
            r, c, s = row_indices[i], col_indices[i], sims[i]
            if r < c:
                pairs.append((index[r], index[c], s))
            elif c < r:
                pairs.append((index[c], index[r], s))

        # Sort and get top-k
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:top_k]

        results = []
        for item1, item2, sim in top_pairs:
            results.append({'item1': item1, 'item2': item2, 'similarity': sim})

        return pd.DataFrame(results)
    else:
        # Full mode
        pairs = []
        for i in range(len(similarity_data)):
            for j in range(i+1, len(similarity_data)):
                pairs.append((
                    similarity_data.index[i],
                    similarity_data.index[j],
                    similarity_data.iloc[i, j]
                ))

        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:top_k]

        results = []
        for item1, item2, sim in top_pairs:
            results.append({'item1': item1, 'item2': item2, 'similarity': sim})

        return pd.DataFrame(results)

def plot_similarity_distribution(similarity_data, output_path, title):
    """Plot distribution of similarity scores"""
    if isinstance(similarity_data, dict):
        similarities = similarity_data['similarities']
    else:
        mask = np.triu(np.ones_like(similarity_data, dtype=bool), k=1)
        similarities = similarity_data.values[mask].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Similarities')
    ax1.axvline(similarities.mean(), color='red', linestyle='--',
                label=f'Mean: {similarities.mean():.3f}')
    ax1.axvline(np.median(similarities), color='green', linestyle='--',
                label=f'Median: {np.median(similarities):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(similarities, vert=True)
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Distribution Summary')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

def main():
    setup_directories()
    print(f"Dataset: {DATASET_NAME}")
    print("=" * 80)

    # Load embeddings
    variant_df, phenotype_df = load_embeddings()

    # =========================================================================
    # 1. PHENOTYPE-PHENOTYPE SIMILARITY
    # =========================================================================
    print("\n[1/3] Phenotype-Phenotype Similarity")
    print("-" * 80)

    pheno_sim = compute_similarity_matrix(phenotype_df, "phenotypes")

    save_similarity_data(
        pheno_sim,
        SIMILARITY_DIR / "phenotype_phenotype_similarity.csv"
    )

    plot_similarity_distribution(
        pheno_sim,
        SIMILARITY_DIR / "phenotype_phenotype_distribution.png",
        f"Phenotype-Phenotype Similarity Distribution"
    )

    pheno_pairs = analyze_top_pairs(pheno_sim, top_k=20)
    pheno_pairs.to_csv(SIMILARITY_DIR / "phenotype_phenotype_top_pairs.csv", index=False)

    # =========================================================================
    # 2. VARIANT-VARIANT SIMILARITY
    # =========================================================================
    print("\n[2/3] Variant-Variant Similarity")
    print("-" * 80)

    variant_sim = compute_similarity_matrix(variant_df, "variants")

    save_similarity_data(
        variant_sim,
        SIMILARITY_DIR / "variant_variant_similarity.csv"
    )

    plot_similarity_distribution(
        variant_sim,
        SIMILARITY_DIR / "variant_variant_distribution.png",
        f"Variant-Variant Similarity Distribution"
    )

    variant_pairs = analyze_top_pairs(variant_sim, top_k=20)
    variant_pairs.to_csv(SIMILARITY_DIR / "variant_variant_top_pairs.csv", index=False)

    # =========================================================================
    # 3. VARIANT-PHENOTYPE CROSS-SIMILARITY
    # =========================================================================
    print("\n[3/3] Variant-Phenotype Cross-Similarity")
    print("-" * 80)

    cross_sim = compute_cross_similarity(variant_df, phenotype_df, top_k=20)

    save_similarity_data(
        cross_sim,
        SIMILARITY_DIR / "variant_phenotype_cross_similarity.csv",
        data_type='cross'
    )

    plot_similarity_distribution(
        cross_sim,
        SIMILARITY_DIR / "variant_phenotype_cross_distribution.png",
        f"Variant-Phenotype Cross-Similarity Distribution"
    )

    cross_pairs = analyze_top_pairs(cross_sim, top_k=20, data_type='cross')
    cross_pairs.to_csv(SIMILARITY_DIR / "variant_phenotype_cross_top_pairs.csv", index=False)

if __name__ == "__main__":
    main()
