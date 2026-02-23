"""
Model Evaluation Module
Compares clustering vs bi-clustering approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare clustering and bi-clustering models"""
    
    def __init__(self):
        """Initialize model comparison"""
        self.results = {}

    @staticmethod
    def _safe_clustering_metrics(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering metrics safely when labels may be degenerate."""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            return {
                'silhouette_score': np.nan,
                'davies_bouldin_score': np.nan,
                'calinski_harabasz_score': np.nan
            }

        return {
            'silhouette_score': silhouette_score(data, labels),
            'davies_bouldin_score': davies_bouldin_score(data, labels),
            'calinski_harabasz_score': calinski_harabasz_score(data, labels)
        }
        
    def evaluate_clustering(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model_name: str = "K-Means Clustering"
    ) -> Dict[str, float]:
        """
        Evaluate clustering model
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data
        labels : np.ndarray
            Cluster labels
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        if len(data) != len(labels):
            raise ValueError(
                f"Length mismatch: data has {len(data)} rows but labels has {len(labels)} entries."
            )

        metric_values = self._safe_clustering_metrics(data, labels)
        
        metrics = {
            'model': model_name,
            'n_clusters': len(np.unique(labels)),
            'silhouette_score': metric_values['silhouette_score'],
            'davies_bouldin_score': metric_values['davies_bouldin_score'],
            'calinski_harabasz_score': metric_values['calinski_harabasz_score']
        }
        
        # Add cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['min_cluster_size'] = counts.min()
        metrics['max_cluster_size'] = counts.max()
        metrics['cluster_balance'] = counts.std() / counts.mean() if counts.mean() > 0 else np.nan
        
        self.results[model_name] = metrics
        
        logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        logger.info(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
        
        return metrics
    
    def evaluate_biclustering(
        self,
        user_artist_matrix: pd.DataFrame,
        user_labels: np.ndarray,
        artist_labels: np.ndarray,
        model_name: str = "Bi-Clustering"
    ) -> Dict[str, float]:
        """
        Evaluate bi-clustering model
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist matrix
        user_labels : np.ndarray
            User cluster labels
        artist_labels : np.ndarray
            Artist cluster labels
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        if len(user_artist_matrix.index) != len(user_labels):
            raise ValueError(
                f"Length mismatch: user_artist_matrix has {len(user_artist_matrix.index)} users "
                f"but user_labels has {len(user_labels)} entries."
            )
        if len(user_artist_matrix.columns) != len(artist_labels):
            raise ValueError(
                f"Length mismatch: user_artist_matrix has {len(user_artist_matrix.columns)} artists "
                f"but artist_labels has {len(artist_labels)} entries."
            )
        
        metrics = {
            'model': model_name,
            'n_user_clusters': len(np.unique(user_labels)),
            'n_artist_clusters': len(np.unique(artist_labels)),
            'n_biclusters': int(max(np.max(user_labels), np.max(artist_labels)) + 1)
        }
        
        # Calculate bi-cluster quality metrics
        bicluster_densities = []
        bicluster_sizes = []
        coverage_users = 0
        coverage_artists = 0
        
        matrix_values = user_artist_matrix.values
        
        for i in range(metrics['n_biclusters']):
            user_mask = user_labels == i
            artist_mask = artist_labels == i
            
            if user_mask.sum() > 0 and artist_mask.sum() > 0:
                # Get submatrix
                submatrix = matrix_values[np.ix_(user_mask, artist_mask)]
                
                # Calculate density (proportion of non-zero entries)
                density = (submatrix > 0).sum() / submatrix.size if submatrix.size > 0 else 0
                bicluster_densities.append(density)
                bicluster_sizes.append(submatrix.size)
                
                coverage_users += user_mask.sum()
                coverage_artists += artist_mask.sum()
        
        metrics['avg_bicluster_density'] = np.mean(bicluster_densities) if bicluster_densities else 0
        metrics['avg_bicluster_size'] = np.mean(bicluster_sizes) if bicluster_sizes else 0
        metrics['user_coverage'] = coverage_users / len(user_labels) if len(user_labels) > 0 else 0
        metrics['artist_coverage'] = coverage_artists / len(artist_labels) if len(artist_labels) > 0 else 0
        
        # User cluster distribution
        unique_users, counts_users = np.unique(user_labels, return_counts=True)
        metrics['user_cluster_sizes'] = dict(zip(unique_users.tolist(), counts_users.tolist()))
        metrics['user_cluster_balance'] = (
            counts_users.std() / counts_users.mean() if counts_users.mean() > 0 else np.nan
        )
        
        self.results[model_name] = metrics
        
        logger.info(f"  Number of bi-clusters: {metrics['n_biclusters']}")
        logger.info(f"  Average bi-cluster density: {metrics['avg_bicluster_density']:.4f}")
        logger.info(f"  User coverage: {metrics['user_coverage']:.2%}")
        logger.info(f"  Artist coverage: {metrics['artist_coverage']:.2%}")
        
        return metrics
    
    def compare_user_clustering(
        self,
        clustering_labels: np.ndarray,
        biclustering_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare user assignments between clustering and bi-clustering
        
        Parameters:
        -----------
        clustering_labels : np.ndarray
            Labels from K-Means clustering
        biclustering_labels : np.ndarray
            User labels from bi-clustering
            
        Returns:
        --------
        dict
            Comparison metrics
        """
        logger.info("Comparing user clustering assignments...")

        if len(clustering_labels) != len(biclustering_labels):
            raise ValueError(
                f"Length mismatch: clustering_labels has {len(clustering_labels)} entries but "
                f"biclustering_labels has {len(biclustering_labels)} entries."
            )
        
        comparison = {
            'adjusted_rand_index': adjusted_rand_score(clustering_labels, biclustering_labels),
            'normalized_mutual_info': normalized_mutual_info_score(clustering_labels, biclustering_labels)
        }
        
        logger.info(f"  Adjusted Rand Index: {comparison['adjusted_rand_index']:.4f}")
        logger.info(f"  Normalized Mutual Information: {comparison['normalized_mutual_info']:.4f}")
        
        self.results['comparison'] = comparison
        
        return comparison
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all models
        
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        logger.info("Creating comparison table...")

        if not self.results:
            return pd.DataFrame({'Model': [], 'Status': []})
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            if model_name != 'comparison':
                row = {'Model': model_name}
                
                # Add relevant metrics
                if 'silhouette_score' in metrics:
                    row['Silhouette Score'] = f"{metrics['silhouette_score']:.4f}"
                    row['Davies-Bouldin Score'] = f"{metrics['davies_bouldin_score']:.4f}"
                    row['Calinski-Harabasz Score'] = f"{metrics['calinski_harabasz_score']:.1f}"
                    row['N Clusters'] = metrics['n_clusters']
                    row['Cluster Balance'] = f"{metrics['cluster_balance']:.4f}"
                
                if 'avg_bicluster_density' in metrics:
                    row['Avg Bi-cluster Density'] = f"{metrics['avg_bicluster_density']:.4f}"
                    row['N Bi-clusters'] = metrics['n_biclusters']
                    row['User Coverage'] = f"{metrics['user_coverage']:.2%}"
                    row['Artist Coverage'] = f"{metrics['artist_coverage']:.2%}"
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)

        if comparison_df.empty:
            return comparison_df

        preferred_order = [
            'Model',
            'Silhouette Score',
            'Davies-Bouldin Score',
            'Calinski-Harabasz Score',
            'N Clusters',
            'Cluster Balance',
            'Avg Bi-cluster Density',
            'N Bi-clusters',
            'User Coverage',
            'Artist Coverage'
        ]

        existing = [col for col in preferred_order if col in comparison_df.columns]
        remaining = [col for col in comparison_df.columns if col not in existing]
        comparison_df = comparison_df[existing + remaining]

        comparison_df = comparison_df.fillna('-')
        
        return comparison_df
    
    def analyze_user_migration(
        self,
        user_ids: np.ndarray,
        clustering_labels: np.ndarray,
        biclustering_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze how users are distributed differently between models
        
        Parameters:
        -----------
        user_ids : np.ndarray
            User IDs
        clustering_labels : np.ndarray
            Labels from clustering
        biclustering_labels : np.ndarray
            Labels from bi-clustering
            
        Returns:
        --------
        pd.DataFrame
            Migration analysis
        """
        logger.info("Analyzing user migration between models...")
        
        migration = pd.DataFrame({
            'userid': user_ids,
            'kmeans_cluster': clustering_labels,
            'bicluster': biclustering_labels
        })
        
        # Create cross-tabulation
        crosstab = pd.crosstab(
            migration['kmeans_cluster'],
            migration['bicluster'],
            margins=True
        )
        
        logger.info("\nUser distribution cross-tabulation:")
        logger.info(f"\n{crosstab}")
        
        return migration, crosstab
    
    def get_summary(self) -> str:
        """
        Get summary of evaluation results
        
        Returns:
        --------
        str
            Summary text
        """
        summary = ["=" * 60]
        summary.append("MODEL COMPARISON SUMMARY")
        summary.append("=" * 60)
        
        for model_name, metrics in self.results.items():
            if model_name == 'comparison':
                summary.append("\nClustering vs Bi-Clustering Agreement:")
                summary.append(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
                summary.append(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
            else:
                summary.append(f"\n{model_name}:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if isinstance(value, float):
                            summary.append(f"  {key}: {value:.4f}")
                        else:
                            summary.append(f"  {key}: {value}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)
