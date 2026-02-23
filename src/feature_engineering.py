"""
Feature Engineering Module
Creates user-artist matrices and other features for clustering
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for clustering from listening data"""
    
    def __init__(self, listening_data: pd.DataFrame, profile_data: pd.DataFrame = None):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        listening_data : pd.DataFrame
            Preprocessed listening history
        profile_data : pd.DataFrame, optional
            User profile data
        """
        self.listening_data = listening_data
        self.profile_data = profile_data
        self.user_artist_matrix = None
        self.user_features = None
        
    def create_user_artist_matrix(self, normalize: bool = True) -> pd.DataFrame:
        """
        Create user-artist play count matrix
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize the matrix
            
        Returns:
        --------
        pd.DataFrame
            User-artist matrix with play counts
        """
        logger.info("Creating user-artist matrix...")
        
        # Create pivot table: users × artists
        user_artist_counts = self.listening_data.groupby(
            ['userid', 'artist_name']
        ).size().reset_index(name='play_count')
        
        # Pivot to matrix format
        self.user_artist_matrix = user_artist_counts.pivot(
            index='userid',
            columns='artist_name',
            values='play_count'
        ).fillna(0)
        
        logger.info(f"Created matrix with shape: {self.user_artist_matrix.shape}")
        logger.info(f"Users: {self.user_artist_matrix.shape[0]}, Artists: {self.user_artist_matrix.shape[1]}")
        
        # Normalize if requested
        if normalize:
            self.user_artist_matrix = self._normalize_matrix(self.user_artist_matrix)
        
        return self.user_artist_matrix
    
    def _normalize_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize user-artist matrix (row-wise: normalize by user)
        
        Parameters:
        -----------
        matrix : pd.DataFrame
            User-artist matrix
            
        Returns:
        --------
        pd.DataFrame
            Normalized matrix
        """
        logger.info("Normalizing user-artist matrix...")
        
        # Normalize by user (each row sums to 1)
        row_sums = matrix.sum(axis=1)
        normalized = matrix.div(row_sums, axis=0)
        
        return normalized
    
    def create_user_features(self) -> pd.DataFrame:
        """
        Create aggregated features for each user
        
        Returns:
        --------
        pd.DataFrame
            User-level features for clustering
        """
        logger.info("Creating user-level features...")
        
        user_features = pd.DataFrame()
        
        # Basic listening patterns
        user_features['total_plays'] = self.listening_data.groupby('userid').size()
        user_features['unique_artists'] = self.listening_data.groupby('userid')['artist_name'].nunique()
        user_features['unique_tracks'] = self.listening_data.groupby('userid')['track_name'].nunique()
        
        # Diversity metrics
        user_features['artist_diversity'] = (
            user_features['unique_artists'] / user_features['total_plays']
        )
        user_features['track_diversity'] = (
            user_features['unique_tracks'] / user_features['total_plays']
        )
        
        # Temporal features (if timestamp available)
        if 'timestamp' in self.listening_data.columns:
            temporal = self.listening_data.groupby('userid')['timestamp'].agg([
                ('listening_span_days', lambda x: (x.max() - x.min()).days),
                ('first_listen', 'min'),
                ('last_listen', 'max')
            ])
            user_features = user_features.join(temporal)
            
            # Listening frequency
            user_features['plays_per_day'] = (
                user_features['total_plays'] / (user_features['listening_span_days'] + 1)
            )
        
        # Top artist concentration (what % of plays go to top artist)
        top_artist_plays = self.listening_data.groupby('userid').apply(
            lambda x: x['artist_name'].value_counts().iloc[0] if len(x) > 0 else 0
        )
        user_features['top_artist_concentration'] = (
            top_artist_plays / user_features['total_plays']
        )
        
        # Add profile features if available
        if self.profile_data is not None:
            user_features = self._add_profile_features(user_features)
        
        self.user_features = user_features
        logger.info(f"Created {user_features.shape[1]} features for {user_features.shape[0]} users")
        
        return user_features
    
    def _add_profile_features(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add profile data to user features
        
        Parameters:
        -----------
        user_features : pd.DataFrame
            Existing user features
            
        Returns:
        --------
        pd.DataFrame
            User features with profile data
        """
        logger.info("Adding profile features...")
        
        profile = self.profile_data.copy()
        profile.set_index('#id' if '#id' in profile.columns else 'userid', inplace=True)
        
        # Add age (fill missing with median)
        if 'age' in profile.columns:
            user_features = user_features.join(profile['age'])
            user_features['age'].fillna(user_features['age'].median(), inplace=True)
        
        # Add gender (one-hot encode)
        if 'gender' in profile.columns:
            user_features = user_features.join(profile['gender'])
            user_features['is_male'] = (user_features['gender'] == 'm').astype(int)
            user_features['is_female'] = (user_features['gender'] == 'f').astype(int)
            user_features.drop('gender', axis=1, inplace=True)
        
        return user_features
    
    def scale_features(
        self,
        features: pd.DataFrame,
        method: str = 'standard'
    ) -> Tuple[np.ndarray, object]:
        """
        Scale features for clustering
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features to scale
        method : str
            Scaling method: 'standard' or 'minmax'
            
        Returns:
        --------
        tuple
            (scaled_features, scaler_object)
        """
        logger.info(f"Scaling features using {method} method...")
        
        # Remove any datetime columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled = scaler.fit_transform(numeric_features)
        
        return scaled, scaler
    
    def get_top_artists_per_user(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N artists for each user
        
        Parameters:
        -----------
        top_n : int
            Number of top artists to return
            
        Returns:
        --------
        pd.DataFrame
            Top artists per user
        """
        logger.info(f"Getting top {top_n} artists per user...")
        
        top_artists = self.listening_data.groupby('userid').apply(
            lambda x: x['artist_name'].value_counts().head(top_n).to_dict()
        )
        
        return top_artists
