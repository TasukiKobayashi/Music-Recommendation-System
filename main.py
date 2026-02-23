"""
CLI entrypoint for Last.fm clustering project.
Run end-to-end pipeline without using the notebook.
"""

import argparse
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    LISTENING_FILE,
    PROFILE_FILE,
    SAMPLE_USERS,
    MIN_ARTIST_PLAYS,
    MIN_USER_PLAYS,
    N_CLUSTERS_RANGE,
    OPTIMAL_CLUSTERS,
    N_BICLUSTERS,
    RANDOM_STATE,
)
from src.data_loader import LastFMDataLoader
from src.feature_engineering import FeatureEngineer
from src.clustering_model import UserClusteringModel
from src.biclustering_model import BiClusteringModel
from src.evaluation import ModelComparison
from src.visualization import ClusteringVisualizer
from src.utils import create_project_structure, save_model, save_results


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Last.fm clustering + bi-clustering pipeline"
    )
    parser.add_argument("--nrows", type=int, default=1_000_000, help="Number of listening rows to load")
    parser.add_argument("--sample-users", type=int, default=SAMPLE_USERS, help="Number of users to sample")
    parser.add_argument("--min-artist-plays", type=int, default=MIN_ARTIST_PLAYS, help="Min plays per artist")
    parser.add_argument("--min-user-plays", type=int, default=MIN_USER_PLAYS, help="Min plays per user")
    parser.add_argument("--k", type=int, default=OPTIMAL_CLUSTERS, help="Number of K-Means clusters")
    parser.add_argument("--biclusters", type=int, default=N_BICLUSTERS, help="Number of bi-clusters")
    parser.add_argument("--skip-optimal-k-search", action="store_true", help="Skip elbow/silhouette K search")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "output"), help="Output directory")
    return parser.parse_args()


def save_figure(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def run_pipeline(args: argparse.Namespace) -> None:
    create_project_structure(BASE_DIR)

    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Starting pipeline...")
    logger.info("Listening file: %s", LISTENING_FILE)
    logger.info("Profile file: %s", PROFILE_FILE)

    loader = LastFMDataLoader(LISTENING_FILE, PROFILE_FILE)

    profile_data = loader.load_profile_data()
    listening_data = loader.load_listening_data(
        nrows=args.nrows,
        sample_users=args.sample_users,
    )
    listening_data = loader.preprocess_data(
        min_artist_plays=args.min_artist_plays,
        min_user_plays=args.min_user_plays,
    )
    stats = loader.get_statistics()

    feature_eng = FeatureEngineer(listening_data=listening_data, profile_data=profile_data)
    user_artist_matrix = feature_eng.create_user_artist_matrix(normalize=False)
    user_features = feature_eng.create_user_features()
    user_features_scaled, scaler = feature_eng.scale_features(user_features, method="standard")

    kmeans_model = UserClusteringModel(random_state=RANDOM_STATE)
    viz = ClusteringVisualizer()

    if not args.skip_optimal_k_search:
        k_metrics = kmeans_model.find_optimal_clusters(
            data=user_features_scaled,
            k_range=N_CLUSTERS_RANGE,
        )
        fig = viz.plot_elbow_curve(k_values=k_metrics["k_values"], inertias=k_metrics["inertia"])
        save_figure(fig, os.path.join(plots_dir, "kmeans_elbow_curve.png"))

        fig = viz.plot_silhouette_scores(
            k_values=k_metrics["k_values"],
            silhouette_scores=k_metrics["silhouette"],
        )
        save_figure(fig, os.path.join(plots_dir, "kmeans_silhouette.png"))

    kmeans_model.fit(data=user_features_scaled, n_clusters=args.k)
    cluster_stats, cluster_sizes = kmeans_model.analyze_clusters(
        data=user_features,
        feature_names=user_features.columns.tolist(),
    )

    fig = viz.plot_cluster_distribution(
        labels=kmeans_model.labels,
        title=f"K-Means Cluster Distribution (K={args.k})",
    )
    save_figure(fig, os.path.join(plots_dir, "kmeans_cluster_distribution.png"))

    data_2d, _ = kmeans_model.reduce_dimensions(data=user_features_scaled, n_components=2)
    fig = viz.plot_2d_clusters(
        data_2d=data_2d,
        labels=kmeans_model.labels,
        title=f"K-Means Clusters Visualization (K={args.k})",
    )
    save_figure(fig, os.path.join(plots_dir, "kmeans_2d_clusters.png"))

    # Feature importance heatmap from robust numeric means
    numeric_features = user_features.select_dtypes(include=[np.number]).copy()
    clustered_numeric = numeric_features.copy()
    clustered_numeric["cluster"] = kmeans_model.labels
    feature_means_numeric = clustered_numeric.groupby("cluster").mean(numeric_only=True)
    feature_variance = feature_means_numeric.var(axis=0).fillna(0)
    feature_variance = feature_variance[feature_variance > 0]
    top_n_effective = min(10, len(feature_variance))
    if top_n_effective > 0:
        top_features = feature_variance.nlargest(top_n_effective).index
        fig, ax = plt.subplots(figsize=(12, 10))
        import seaborn as sns

        sns.heatmap(
            feature_means_numeric[top_features].T,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Mean Value"},
            ax=ax,
        )
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(
            f"Top {top_n_effective} Discriminative Features Across Clusters",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, os.path.join(plots_dir, "kmeans_feature_importance.png"))

    bicluster_model = BiClusteringModel(random_state=RANDOM_STATE)
    bicluster_model.fit(user_artist_matrix=user_artist_matrix, n_clusters=args.biclusters)
    user_labels_bi, artist_labels_bi = bicluster_model.get_biclusters()
    bicluster_stats = bicluster_model.analyze_biclusters(user_artist_matrix=user_artist_matrix)

    fig = viz.plot_bicluster_statistics(
        bicluster_stats=bicluster_stats,
        title=f"Bi-Cluster Statistics (N={args.biclusters})",
    )
    save_figure(fig, os.path.join(plots_dir, "bicluster_statistics.png"))

    fig = viz.plot_bicluster_heatmap(
        user_artist_matrix=user_artist_matrix,
        user_labels=user_labels_bi,
        artist_labels=artist_labels_bi,
        sample_size=150,
        title="User-Artist Bi-Clustering Heatmap (Reordered)",
    )
    save_figure(fig, os.path.join(plots_dir, "bicluster_heatmap.png"))

    comparison = ModelComparison()
    kmeans_metrics = comparison.evaluate_clustering(
        data=user_features_scaled,
        labels=kmeans_model.labels,
        model_name="K-Means Clustering",
    )
    bicluster_metrics = comparison.evaluate_biclustering(
        user_artist_matrix=user_artist_matrix,
        user_labels=user_labels_bi,
        artist_labels=artist_labels_bi,
        model_name="Bi-Clustering",
    )
    user_comparison = comparison.compare_user_clustering(
        clustering_labels=kmeans_model.labels,
        biclustering_labels=user_labels_bi,
    )
    migration_df, crosstab = comparison.analyze_user_migration(
        user_ids=user_artist_matrix.index.values,
        clustering_labels=kmeans_model.labels,
        biclustering_labels=user_labels_bi,
    )

    comparison_summary = comparison.get_summary()
    with open(os.path.join(results_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write(comparison_summary)

    # Save tabular results
    cluster_sizes.to_csv(os.path.join(results_dir, "cluster_sizes.csv"), index=True)
    bicluster_stats.to_csv(os.path.join(results_dir, "bicluster_stats.csv"), index=False)
    migration_df.to_csv(os.path.join(results_dir, "user_cluster_migration.csv"), index=False)
    crosstab.to_csv(os.path.join(results_dir, "user_cluster_crosstab.csv"))

    user_assignments = pd.DataFrame(
        {
            "userid": user_artist_matrix.index,
            "kmeans_cluster": kmeans_model.labels,
            "bicluster": user_labels_bi,
        }
    )
    user_assignments.to_csv(os.path.join(results_dir, "user_cluster_assignments.csv"), index=False)

    save_model(kmeans_model, os.path.join(models_dir, "kmeans_model.pkl"))
    save_model(bicluster_model, os.path.join(models_dir, "bicluster_model.pkl"))
    save_model(scaler, os.path.join(models_dir, "feature_scaler.pkl"))

    all_results = {
        "dataset_stats": stats,
        "kmeans_metrics": kmeans_metrics,
        "bicluster_metrics": bicluster_metrics,
        "comparison_metrics": user_comparison,
        "optimal_k": args.k,
        "n_biclusters": args.biclusters,
    }
    save_results(all_results, os.path.join(results_dir, "analysis_results.json"))

    logger.info("Pipeline complete.")
    logger.info("Results saved in: %s", output_dir)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
