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

This analysis uses ALL 50 dimensions (not just the first 2-3 shown in PCA plots),
capturing the full semantic information in the embeddings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

# Import configuration
from config import DATASET_NAME, DATASET_DIR, EMBEDDINGS_DIR

def load_embeddings(embeddings_dir):
    """Load variant and phenotype embeddings from CSV files"""
    # Find embedding files
    variant_files = list(embeddings_dir.glob('*_variant_embeddings.csv'))
    phenotype_files = list(embeddings_dir.glob('*_phenotype_embeddings.csv'))

    if not variant_files or not phenotype_files:
        print(f"Error: No embedding files found in {embeddings_dir}")
        print("Run 3_create_embeddings.py first!")
        sys.exit(1)

    # Use the first (or most recent) embedding file
    variant_file = variant_files[0]
    phenotype_file = phenotype_files[0]

    print(f"Loading embeddings from:")
    print(f"  - {variant_file.name}")
    print(f"  - {phenotype_file.name}")

    variant_df = pd.read_csv(variant_file, index_col=0)
    phenotype_df = pd.read_csv(phenotype_file, index_col=0)

    print(f"\nVariant embeddings shape: {variant_df.shape} (variants x dimensions)")
    print(f"Phenotype embeddings shape: {phenotype_df.shape} (phenotypes x dimensions)")

    return variant_df, phenotype_df, variant_file.stem

def compute_similarity_matrix(embeddings_df, name):
    """Compute pairwise cosine similarity matrix"""
    print(f"\nComputing cosine similarity for {name}...")

    # Compute cosine similarity between all pairs
    similarity_matrix = cosine_similarity(embeddings_df.values)

    # Convert to DataFrame with labels
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=embeddings_df.index,
        columns=embeddings_df.index
    )

    print(f"  Similarity matrix shape: {similarity_df.shape}")
    print(f"  Similarity range: [{similarity_df.values.min():.3f}, {similarity_df.values.max():.3f}]")

    return similarity_df

def find_most_similar(similarity_df, query_item, top_k=10):
    """Find the k most similar items to a query item"""
    if query_item not in similarity_df.index:
        print(f"Error: '{query_item}' not found in embeddings")
        return None

    # Get similarities for this item (excluding itself)
    similarities = similarity_df[query_item].copy()
    similarities = similarities.drop(query_item)

    # Sort by similarity (descending)
    top_similar = similarities.nlargest(top_k)

    return top_similar

def plot_similarity_heatmap(similarity_df, output_path, title, max_items=50):
    """Plot similarity matrix as a heatmap

    Args:
        similarity_df: Pairwise similarity matrix
        output_path: Where to save the plot
        title: Plot title
        max_items: Maximum number of items to show (for readability)
    """
    # If too many items, sample or take top variance
    if len(similarity_df) > max_items:
        print(f"  Too many items ({len(similarity_df)}), showing top {max_items} by variance...")
        # Select items with highest variance in similarity (most interesting)
        variances = similarity_df.var(axis=1)
        top_items = variances.nlargest(max_items).index
        similarity_df = similarity_df.loc[top_items, top_items]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(
        similarity_df,
        cmap='RdYlBu_r',  # Red (high similarity) to Blue (low similarity)
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_similarity_distribution(similarity_df, output_path, title):
    """Plot distribution of similarity scores"""
    # Get upper triangle (excluding diagonal) to avoid duplicates
    mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
    similarities = similarity_df.values[mask].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Pairwise Similarities')
    ax1.axvline(similarities.mean(), color='red', linestyle='--',
                label=f'Mean: {similarities.mean():.3f}')
    ax1.axvline(np.median(similarities), color='green', linestyle='--',
                label=f'Median: {np.median(similarities):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(similarities, vert=True)
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Similarity Distribution Summary')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def analyze_top_similar_pairs(similarity_df, top_k=20):
    """Find and display the most similar pairs"""
    print(f"\nTop {top_k} most similar pairs:")
    print("=" * 80)

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)

    # Get all pairs with their similarities
    pairs = []
    for i in range(len(similarity_df)):
        for j in range(i+1, len(similarity_df)):
            pairs.append((
                similarity_df.index[i],
                similarity_df.index[j],
                similarity_df.iloc[i, j]
            ))

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Display top pairs
    results = []
    for rank, (item1, item2, sim) in enumerate(pairs[:top_k], 1):
        print(f"{rank:2d}. {item1:30s} ↔ {item2:30s}  |  Similarity: {sim:.4f}")
        results.append({'rank': rank, 'item1': item1, 'item2': item2, 'similarity': sim})

    return pd.DataFrame(results)

def analyze_query_similarities(similarity_df, query_items, top_k=10):
    """Analyze similarities for specific query items"""
    results = {}

    for query in query_items:
        if query in similarity_df.index:
            print(f"\nTop {top_k} most similar to '{query}':")
            print("-" * 80)
            similar = find_most_similar(similarity_df, query, top_k)

            for rank, (item, sim) in enumerate(similar.items(), 1):
                print(f"{rank:2d}. {item:40s}  |  Similarity: {sim:.4f}")

            results[query] = similar
        else:
            print(f"\nWarning: '{query}' not found in embeddings")

    return results

def save_similarity_matrix(similarity_df, output_path):
    """Save full similarity matrix to CSV"""
    similarity_df.to_csv(output_path)
    print(f"\nSaved full similarity matrix to: {output_path}")

def main():
    print(f"Dataset: {DATASET_NAME}")
    print("=" * 80)

    # Load embeddings
    variant_df, phenotype_df, matrix_name = load_embeddings(EMBEDDINGS_DIR)

    # Create output directory for similarity analysis
    similarity_dir = EMBEDDINGS_DIR / "similarity_analysis"
    similarity_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {similarity_dir.absolute()}")
    print("=" * 80)

    # =========================================================================
    # PHENOTYPE SIMILARITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHENOTYPE SIMILARITY ANALYSIS")
    print("=" * 80)

    pheno_sim = compute_similarity_matrix(phenotype_df, "phenotypes")

    # Save full similarity matrix
    save_similarity_matrix(
        pheno_sim,
        similarity_dir / f"{matrix_name}_phenotype_similarity.csv"
    )

    # Plot heatmap
    print("\nCreating phenotype similarity visualizations...")
    plot_similarity_heatmap(
        pheno_sim,
        similarity_dir / f"{matrix_name}_phenotype_similarity_heatmap.png",
        f"Phenotype Similarity Heatmap ({DATASET_NAME})\nCosine similarity using all {phenotype_df.shape[1]} embedding dimensions",
        max_items=50
    )

    # Plot distribution
    plot_similarity_distribution(
        pheno_sim,
        similarity_dir / f"{matrix_name}_phenotype_similarity_distribution.png",
        f"Phenotype Similarity Distribution ({DATASET_NAME})"
    )

    # Analyze top similar pairs
    top_pairs = analyze_top_similar_pairs(pheno_sim, top_k=20)
    top_pairs.to_csv(
        similarity_dir / f"{matrix_name}_phenotype_top_pairs.csv",
        index=False
    )

    # Example queries (if you want to search for specific phenotypes)
    # Modify this list based on phenotypes in your dataset
    example_phenotypes = list(phenotype_df.index[:3])  # First 3 as examples
    print("\n" + "=" * 80)
    print("EXAMPLE PHENOTYPE QUERIES")
    print("=" * 80)
    analyze_query_similarities(pheno_sim, example_phenotypes, top_k=10)

    # =========================================================================
    # VARIANT SIMILARITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("VARIANT SIMILARITY ANALYSIS")
    print("=" * 80)

    variant_sim = compute_similarity_matrix(variant_df, "variants")

    # Save full similarity matrix
    save_similarity_matrix(
        variant_sim,
        similarity_dir / f"{matrix_name}_variant_similarity.csv"
    )

    # Plot heatmap (sample for visualization due to large size)
    print("\nCreating variant similarity visualizations...")
    plot_similarity_heatmap(
        variant_sim,
        similarity_dir / f"{matrix_name}_variant_similarity_heatmap.png",
        f"Variant Similarity Heatmap ({DATASET_NAME})\nCosine similarity using all {variant_df.shape[1]} embedding dimensions",
        max_items=50
    )

    # Plot distribution
    plot_similarity_distribution(
        variant_sim,
        similarity_dir / f"{matrix_name}_variant_similarity_distribution.png",
        f"Variant Similarity Distribution ({DATASET_NAME})"
    )

    # Analyze top similar variant pairs
    top_variant_pairs = analyze_top_similar_pairs(variant_sim, top_k=20)
    top_variant_pairs.to_csv(
        similarity_dir / f"{matrix_name}_variant_top_pairs.csv",
        index=False
    )

    # Example variant queries
    example_variants = list(variant_df.index[:3])  # First 3 as examples
    print("\n" + "=" * 80)
    print("EXAMPLE VARIANT QUERIES")
    print("=" * 80)
    analyze_query_similarities(variant_sim, example_variants, top_k=10)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {similarity_dir.absolute()}")
    print("\nGenerated files:")
    print("  - *_similarity.csv: Full similarity matrices")
    print("  - *_similarity_heatmap.png: Heatmap visualizations")
    print("  - *_similarity_distribution.png: Distribution plots")
    print("  - *_top_pairs.csv: Most similar pairs")
    print("\nNext steps:")
    print("  1. Load similarity matrices to query specific phenotypes/variants")
    print("  2. Use cosine similarity for semantic search across all dimensions")
    print("  3. Cluster analysis based on similarity thresholds")
    print("=" * 80)

if __name__ == "__main__":
    main()
