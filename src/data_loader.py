"""
Data Loading and Preprocessing Module
Handles loading and initial preprocessing of Last.fm dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LastFMDataLoader:
    """Load and preprocess Last.fm dataset"""
    
    def __init__(self, listening_file: str, profile_file: str):
        """
        Initialize data loader
        
        Parameters:
        -----------
        listening_file : str
            Path to listening history TSV file
        profile_file : str
            Path to user profile TSV file
        """
        self.listening_file = listening_file
        self.profile_file = profile_file
        self.listening_data = None
        self.profile_data = None
        
    def load_profile_data(self) -> pd.DataFrame:
        """
        Load user profile data
        
        Returns:
        --------
        pd.DataFrame
            User profile data with demographics
        """
        logger.info("Loading user profile data...")
        self.profile_data = pd.read_csv(
            self.profile_file, 
            sep='\t',
            encoding='utf-8'
        )
        logger.info(f"Loaded {len(self.profile_data)} user profiles")
        return self.profile_data
    
    def load_listening_data(
        self, 
        nrows: Optional[int] = None,
        sample_users: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load listening history data
        
        Parameters:
        -----------
        nrows : int, optional
            Number of rows to load (for memory efficiency)
        sample_users : int, optional
            Number of users to sample
            
        Returns:
        --------
        pd.DataFrame
            Listening history data
        """
        logger.info("Loading listening history data...")
        
        # Column names based on README
        columns = ['userid', 'timestamp', 'artist_mbid', 'artist_name', 
                   'track_mbid', 'track_name']
        
        # Load data in chunks if needed
        if nrows:
            self.listening_data = pd.read_csv(
                self.listening_file,
                sep='\t',
                names=columns,
                encoding='utf-8',
                nrows=nrows
            )
        else:
            # Load in chunks and combine
            chunk_size = 100000
            chunks = []
            for chunk in pd.read_csv(
                self.listening_file,
                sep='\t',
                names=columns,
                encoding='utf-8',
                chunksize=chunk_size
            ):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= 1000000:  # Limit to 1M rows
                    break
            self.listening_data = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"Loaded {len(self.listening_data)} listening records")
        
        # Sample users if requested
        if sample_users:
            self.listening_data = self._sample_users(sample_users)
            
        return self.listening_data
    
    def _sample_users(self, n_users: int) -> pd.DataFrame:
        """
        Sample a subset of users
        
        Parameters:
        -----------
        n_users : int
            Number of users to sample
            
        Returns:
        --------
        pd.DataFrame
            Sampled listening data
        """
        logger.info(f"Sampling {n_users} users...")
        unique_users = self.listening_data['userid'].unique()
        
        if len(unique_users) > n_users:
            sampled_users = np.random.choice(unique_users, n_users, replace=False)
            sampled_data = self.listening_data[
                self.listening_data['userid'].isin(sampled_users)
            ]
            logger.info(f"Sampled {len(sampled_data)} records from {n_users} users")
            return sampled_data
        
        return self.listening_data
    
    def preprocess_data(
        self,
        min_artist_plays: int = 5,
        min_user_plays: int = 20
    ) -> pd.DataFrame:
        """
        Preprocess listening data by filtering low-frequency users and artists
        
        Parameters:
        -----------
        min_artist_plays : int
            Minimum number of plays for an artist
        min_user_plays : int
            Minimum number of plays for a user
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed listening data
        """
        if self.listening_data is None:
            raise ValueError("Load listening data first using load_listening_data()")
        
        logger.info("Preprocessing data...")
        df = self.listening_data.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Count plays per artist
        artist_counts = df['artist_name'].value_counts()
        popular_artists = artist_counts[artist_counts >= min_artist_plays].index
        df = df[df['artist_name'].isin(popular_artists)]
        logger.info(f"Filtered to {len(popular_artists)} artists with >={min_artist_plays} plays")
        
        # Count plays per user
        user_counts = df['userid'].value_counts()
        active_users = user_counts[user_counts >= min_user_plays].index
        df = df[df['userid'].isin(active_users)]
        logger.info(f"Filtered to {len(active_users)} users with >={min_user_plays} plays")
        
        self.listening_data = df
        return df
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics
        
        Returns:
        --------
        dict
            Dictionary with dataset statistics
        """
        if self.listening_data is None:
            raise ValueError("Load listening data first")
        
        stats = {
            'total_records': len(self.listening_data),
            'unique_users': self.listening_data['userid'].nunique(),
            'unique_artists': self.listening_data['artist_name'].nunique(),
            'unique_tracks': self.listening_data['track_name'].nunique(),
            'date_range': (
                self.listening_data['timestamp'].min(),
                self.listening_data['timestamp'].max()
            ) if 'timestamp' in self.listening_data.columns else None
        }
        
        return stats
