"""
Clustering Model Module
Implements traditional clustering (K-Means) for user segmentation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import logging
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserClusteringModel:
    """K-Means clustering model for user segmentation"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize clustering model
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.n_clusters = None
        self.metrics = {}
        
    def find_optimal_clusters(
        self,
        data: np.ndarray,
        k_range: range = range(2, 11)
    ) -> Dict[str, list]:
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data
        k_range : range
            Range of K values to try
            
        Returns:
        --------
        dict
            Dictionary with metrics for each K value
        """
        logger.info(f"Finding optimal clusters in range {k_range}...")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        for k in k_range:
            # Fit K-Means
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(data, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(data, labels))
            
            logger.info(
                f"K={k}: Silhouette={silhouette_scores[-1]:.3f}, "
                f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}"
            )
        
        metrics = {
            'k_values': list(k_range),
            'inertia': inertias,
            'silhouette': silhouette_scores,
            'davies_bouldin': davies_bouldin_scores,
            'calinski_harabasz': calinski_harabasz_scores
        }
        
        # Find optimal K (highest silhouette score)
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        logger.info(f"Optimal K (by silhouette score): {optimal_k}")
        
        self.metrics = metrics
        return metrics
    
    def fit(self, data: np.ndarray, n_clusters: int) -> 'UserClusteringModel':
        """
        Fit K-Means clustering model
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        self
        """
        logger.info(f"Fitting K-Means with {n_clusters} clusters...")
        
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.model.fit_predict(data)
        
        # Calculate final metrics
        self.metrics['final_silhouette'] = silhouette_score(data, self.labels)
        self.metrics['final_davies_bouldin'] = davies_bouldin_score(data, self.labels)
        self.metrics['final_calinski_harabasz'] = calinski_harabasz_score(data, self.labels)
        self.metrics['inertia'] = self.model.inertia_
        
        logger.info(f"Clustering complete. Silhouette score: {self.metrics['final_silhouette']:.3f}")
        
        return self

    def fit_realtime(
        self,
        data: np.ndarray,
        n_clusters: int,
        epochs: int = 25,
        batch_size: int = 256,
        display_every: int = 1,
        random_state: int = None
    ) -> Dict[str, List[float]]:
        """
        Fit clustering model with real-time epoch visualization.

        Notes:
        ------
        This uses MiniBatchKMeans and calls partial_fit each epoch so progress
        can be visualized live in notebooks.

        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data.
        n_clusters : int
            Number of clusters.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size used at each epoch.
        display_every : int
            Update plot every N epochs.
        random_state : int
            Optional override random state.

        Returns:
        --------
        dict
            Training history with epoch, inertia, and center_shift.
        """
        logger.info(
            f"Fitting MiniBatchKMeans in realtime with {n_clusters} clusters for {epochs} epochs..."
        )

        try:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display
            plotting_available = True
        except Exception:
            plotting_available = False

        self.n_clusters = n_clusters
        rs = self.random_state if random_state is None else random_state

        mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=rs,
            batch_size=batch_size,
            n_init=1,
            max_iter=1,
            init='k-means++',
            reassignment_ratio=0.01
        )

        n_samples = data.shape[0]
        history = {
            'epoch': [],
            'inertia': [],
            'center_shift': []
        }

        prev_centers = None
        rng = np.random.default_rng(rs)

        for epoch in range(1, epochs + 1):
            batch_idx = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            batch = data[batch_idx]
            mbk.partial_fit(batch)

            labels_epoch = mbk.predict(data)
            inertia_epoch = float(np.sum((data - mbk.cluster_centers_[labels_epoch]) ** 2))

            if prev_centers is None:
                shift = 0.0
            else:
                shift = float(np.linalg.norm(mbk.cluster_centers_ - prev_centers))

            prev_centers = mbk.cluster_centers_.copy()

            history['epoch'].append(epoch)
            history['inertia'].append(inertia_epoch)
            history['center_shift'].append(shift)

            if plotting_available and (epoch % max(display_every, 1) == 0 or epoch == epochs):
                clear_output(wait=True)
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                axes[0].plot(history['epoch'], history['inertia'], color='tab:blue', marker='o')
                axes[0].set_title('Realtime Training: Inertia')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Inertia')
                axes[0].grid(alpha=0.3)

                axes[1].plot(history['epoch'], history['center_shift'], color='tab:orange', marker='o')
                axes[1].set_title('Realtime Training: Center Shift')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Shift (L2)')
                axes[1].grid(alpha=0.3)

                fig.suptitle(f'MiniBatchKMeans Progress (Epoch {epoch}/{epochs})', fontsize=12)
                plt.tight_layout()
                display(fig)
                plt.close(fig)

            logger.info(
                f"Epoch {epoch:03d}/{epochs}: inertia={inertia_epoch:.2f}, center_shift={shift:.6f}"
            )

        self.model = mbk
        self.labels = mbk.predict(data)

        self.metrics['final_silhouette'] = silhouette_score(data, self.labels)
        self.metrics['final_davies_bouldin'] = davies_bouldin_score(data, self.labels)
        self.metrics['final_calinski_harabasz'] = calinski_harabasz_score(data, self.labels)
        self.metrics['inertia'] = history['inertia'][-1]
        self.metrics['training_history'] = history

        logger.info(
            f"Realtime clustering complete. Final silhouette: {self.metrics['final_silhouette']:.3f}"
        )

        return history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data
            
        Returns:
        --------
        np.ndarray
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(data)
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids
        
        Returns:
        --------
        np.ndarray
            Cluster centers
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.cluster_centers_
    
    def analyze_clusters(
        self,
        data: pd.DataFrame,
        feature_names: list
    ) -> pd.DataFrame:
        """
        Analyze cluster characteristics
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original feature data (unscaled)
        feature_names : list
            Names of features
            
        Returns:
        --------
        pd.DataFrame
            Cluster statistics
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        logger.info("Analyzing cluster characteristics...")
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = self.labels
        
        # Calculate cluster statistics
        cluster_stats = data_with_clusters.groupby('cluster').agg(['mean', 'std', 'count'])
        
        # Get cluster sizes
        cluster_sizes = pd.DataFrame({
            'size': data_with_clusters['cluster'].value_counts().sort_index(),
            'percentage': data_with_clusters['cluster'].value_counts(normalize=True).sort_index() * 100
        })
        
        logger.info("\nCluster sizes:")
        for idx, row in cluster_sizes.iterrows():
            logger.info(f"  Cluster {idx}: {row['size']} users ({row['percentage']:.1f}%)")
        
        return cluster_stats, cluster_sizes
    
    def reduce_dimensions(
        self,
        data: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, PCA]:
        """
        Reduce dimensions using PCA for visualization
        
        Parameters:
        -----------
        data : np.ndarray
            High-dimensional data
        n_components : int
            Number of components
            
        Returns:
        --------
        tuple
            (reduced_data, pca_object)
        """
        logger.info(f"Reducing dimensions to {n_components}D using PCA...")
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced_data = pca.fit_transform(data)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"Variance explained: {variance_explained:.2%}")
        
        return reduced_data, pca
