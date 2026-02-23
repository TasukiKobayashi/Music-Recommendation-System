"""
Bi-Clustering Model Module
Implements bi-clustering (Spectral Co-Clustering) for simultaneous user-artist clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.metrics import consensus_score
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiClusteringModel:
    """
    Bi-clustering model for simultaneous user and artist clustering
    Uses Spectral Co-Clustering to find coherent user-artist groups
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize bi-clustering model
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.user_labels = None
        self.artist_labels = None
        self.n_clusters = None
        self.metrics = {}
        
    def fit(
        self,
        user_artist_matrix: pd.DataFrame,
        n_clusters: int = 4
    ) -> 'BiClusteringModel':
        """
        Fit bi-clustering model on user-artist matrix
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist play count matrix
        n_clusters : int
            Number of bi-clusters
            
        Returns:
        --------
        self
        """
        logger.info(f"Fitting bi-clustering model with {n_clusters} bi-clusters...")
        
        self.n_clusters = n_clusters
        
        # Use Spectral Co-Clustering
        self.model = SpectralCoclustering(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        # Fit on the matrix
        self.model.fit(user_artist_matrix.values)
        
        # Get cluster labels
        self.user_labels = self.model.row_labels_
        self.artist_labels = self.model.column_labels_
        
        logger.info("Bi-clustering complete.")
        logger.info(f"User clusters: {np.unique(self.user_labels)}")
        logger.info(f"Artist clusters: {np.unique(self.artist_labels)}")
        
        return self
    
    def get_biclusters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bi-cluster assignments
        
        Returns:
        --------
        tuple
            (user_labels, artist_labels)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.user_labels, self.artist_labels
    
    def analyze_biclusters(
        self,
        user_artist_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze bi-cluster characteristics
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            Original user-artist matrix
            
        Returns:
        --------
        pd.DataFrame
            Bi-cluster statistics
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        logger.info("Analyzing bi-cluster characteristics...")
        
        stats = []
        
        for i in range(self.n_clusters):
            # Find users and artists in this bi-cluster
            user_mask = self.user_labels == i
            artist_mask = self.artist_labels == i
            
            # Get submatrix
            submatrix = user_artist_matrix.iloc[user_mask, artist_mask]
            
            stat = {
                'bicluster': i,
                'n_users': user_mask.sum(),
                'n_artists': artist_mask.sum(),
                'total_plays': submatrix.sum().sum(),
                'avg_plays_per_user': submatrix.sum(axis=1).mean(),
                'avg_plays_per_artist': submatrix.sum(axis=0).mean(),
                'density': (submatrix > 0).sum().sum() / (submatrix.shape[0] * submatrix.shape[1])
                    if submatrix.shape[0] * submatrix.shape[1] > 0 else 0
            }
            stats.append(stat)
        
        stats_df = pd.DataFrame(stats)
        
        logger.info("\nBi-cluster statistics:")
        for _, row in stats_df.iterrows():
            logger.info(
                f"  Bi-cluster {int(row['bicluster'])}: "
                f"{int(row['n_users'])} users, {int(row['n_artists'])} artists, "
                f"density={row['density']:.3f}"
            )
        
        return stats_df
    
    def get_top_artists_per_bicluster(
        self,
        user_artist_matrix: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[int, list]:
        """
        Get top artists for each bi-cluster
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist matrix
        top_n : int
            Number of top artists to return
            
        Returns:
        --------
        dict
            Dictionary mapping bi-cluster to top artists
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        logger.info(f"Getting top {top_n} artists per bi-cluster...")
        
        top_artists = {}
        
        for i in range(self.n_clusters):
            # Find users and artists in this bi-cluster
            user_mask = self.user_labels == i
            artist_mask = self.artist_labels == i
            
            # Get submatrix and sum plays per artist
            submatrix = user_artist_matrix.iloc[user_mask, artist_mask]
            artist_plays = submatrix.sum(axis=0).sort_values(ascending=False)
            
            top_artists[i] = artist_plays.head(top_n).to_dict()
        
        return top_artists
    
    def get_user_bicluster_membership(
        self,
        user_artist_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get bi-cluster membership for each user
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist matrix
            
        Returns:
        --------
        pd.DataFrame
            User bi-cluster assignments
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        user_clusters = pd.DataFrame({
            'userid': user_artist_matrix.index,
            'bicluster': self.user_labels
        })
        
        return user_clusters
    
    def get_artist_bicluster_membership(
        self,
        user_artist_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get bi-cluster membership for each artist
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist matrix
            
        Returns:
        --------
        pd.DataFrame
            Artist bi-cluster assignments
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        artist_clusters = pd.DataFrame({
            'artist': user_artist_matrix.columns,
            'bicluster': self.artist_labels
        })
        
        return artist_clusters
    
    def get_bicluster_matrix(
        self,
        user_artist_matrix: pd.DataFrame,
        bicluster_id: int
    ) -> pd.DataFrame:
        """
        Get the submatrix for a specific bi-cluster
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            Original user-artist matrix
        bicluster_id : int
            Bi-cluster ID
            
        Returns:
        --------
        pd.DataFrame
            Submatrix for the bi-cluster
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        user_mask = self.user_labels == bicluster_id
        artist_mask = self.artist_labels == bicluster_id
        
        submatrix = user_artist_matrix.iloc[user_mask, artist_mask]
        
        return submatrix
