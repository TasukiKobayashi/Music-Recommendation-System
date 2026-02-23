"""
Visualization Module
Creates visualizations for clustering and bi-clustering results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringVisualizer:
    """Visualize clustering and bi-clustering results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_elbow_curve(
        self,
        k_values: list,
        inertias: list,
        title: str = "Elbow Method for Optimal K"
    ) -> plt.Figure:
        """
        Plot elbow curve
        
        Parameters:
        -----------
        k_values : list
            K values tested
        inertias : list
            Inertia values for each K
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(k_values, inertias, marker='o', markersize=10, linewidth=2)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette_scores(
        self,
        k_values: list,
        silhouette_scores: list,
        title: str = "Silhouette Score vs Number of Clusters"
    ) -> plt.Figure:
        """
        Plot silhouette scores
        
        Parameters:
        -----------
        k_values : list
            K values tested
        silhouette_scores : list
            Silhouette scores for each K
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(k_values, silhouette_scores, marker='s', markersize=10, 
                linewidth=2, color='green')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best score
        best_idx = np.argmax(silhouette_scores)
        ax.axvline(k_values[best_idx], color='red', linestyle='--', 
                   label=f'Optimal K={k_values[best_idx]}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        title: str = "Cluster Size Distribution"
    ) -> plt.Figure:
        """
        Plot cluster size distribution
        
        Parameters:
        -----------
        labels : np.ndarray
            Cluster labels
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique, counts = np.unique(labels, return_counts=True)
        
        colors = sns.color_palette("husl", len(unique))
        ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(unique)
        
        # Add value labels on bars
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            ax.text(cluster, count + max(counts)*0.01, str(count), 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_clusters(
        self,
        data_2d: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Visualization (2D PCA)",
        xlabel: str = "Principal Component 1",
        ylabel: str = "Principal Component 2"
    ) -> plt.Figure:
        """
        Plot 2D cluster visualization
        
        Parameters:
        -----------
        data_2d : np.ndarray
            2D projected data
        labels : np.ndarray
            Cluster labels
        title : str
            Plot title
        xlabel, ylabel : str
            Axis labels
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_clusters = len(np.unique(labels))
        colors = sns.color_palette("husl", n_clusters)
        
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(
                data_2d[mask, 0], 
                data_2d[mask, 1],
                c=[colors[i]], 
                label=f'Cluster {i}',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        cluster_stats: pd.DataFrame,
        feature_names: list,
        top_n: int = 10
    ) -> plt.Figure:
        """
        Plot feature importance across clusters
        
        Parameters:
        -----------
        cluster_stats : pd.DataFrame
            Cluster statistics
        feature_names : list
            Feature names
        top_n : int
            Number of top features to show
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Get mean values for each feature across clusters
        feature_means = cluster_stats.xs('mean', level=1, axis=1)
        feature_means = feature_means.select_dtypes(include=[np.number])

        if feature_means.empty:
            raise ValueError("No numeric features available for feature importance plot.")
        
        # Select top features by variance
        feature_variance = feature_means.var(numeric_only=True)
        top_features = feature_variance.nlargest(min(top_n, len(feature_variance))).index
        
        # Plot heatmap
        sns.heatmap(
            feature_means[top_features].T,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean Value'},
            ax=ax
        )
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Discriminative Features Across Clusters', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_bicluster_heatmap(
        self,
        user_artist_matrix: pd.DataFrame,
        user_labels: np.ndarray,
        artist_labels: np.ndarray,
        sample_size: int = 100,
        title: str = "Bi-Clustering Heatmap"
    ) -> plt.Figure:
        """
        Plot bi-clustering heatmap
        
        Parameters:
        -----------
        user_artist_matrix : pd.DataFrame
            User-artist matrix
        user_labels : np.ndarray
            User cluster labels
        artist_labels : np.ndarray
            Artist cluster labels
        sample_size : int
            Number of users/artists to sample for visualization
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        # Sort by cluster labels
        user_order = np.argsort(user_labels)
        artist_order = np.argsort(artist_labels)
        
        # Sample if too large
        if len(user_order) > sample_size:
            user_sample = np.random.choice(user_order, sample_size, replace=False)
            user_order = np.sort(user_sample)
        
        if len(artist_order) > sample_size:
            artist_sample = np.random.choice(artist_order, sample_size, replace=False)
            artist_order = np.sort(artist_sample)
        
        # Reorder matrix
        ordered_matrix = user_artist_matrix.iloc[user_order, artist_order]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot heatmap
        sns.heatmap(
            ordered_matrix,
            cmap='Blues',
            cbar_kws={'label': 'Play Count'},
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )
        
        ax.set_xlabel('Artists (sorted by bi-cluster)', fontsize=12)
        ax.set_ylabel('Users (sorted by bi-cluster)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_bicluster_statistics(
        self,
        bicluster_stats: pd.DataFrame,
        title: str = "Bi-Cluster Statistics"
    ) -> plt.Figure:
        """
        Plot bi-cluster statistics
        
        Parameters:
        -----------
        bicluster_stats : pd.DataFrame
            Bi-cluster statistics
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Number of users and artists
        ax1 = axes[0, 0]
        x = bicluster_stats['bicluster']
        width = 0.35
        ax1.bar(x - width/2, bicluster_stats['n_users'], width, 
                label='Users', color='skyblue', edgecolor='black')
        ax1.bar(x + width/2, bicluster_stats['n_artists'], width, 
                label='Artists', color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Bi-cluster', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Users and Artists per Bi-cluster', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Density
        ax2 = axes[0, 1]
        colors = sns.color_palette("husl", len(bicluster_stats))
        ax2.bar(bicluster_stats['bicluster'], bicluster_stats['density'], 
                color=colors, edgecolor='black')
        ax2.set_xlabel('Bi-cluster', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Bi-cluster Density', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average plays per user
        ax3 = axes[1, 0]
        ax3.bar(bicluster_stats['bicluster'], bicluster_stats['avg_plays_per_user'], 
                color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Bi-cluster', fontsize=11)
        ax3.set_ylabel('Avg Plays', fontsize=11)
        ax3.set_title('Average Plays per User', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Total plays
        ax4 = axes[1, 1]
        ax4.bar(bicluster_stats['bicluster'], bicluster_stats['total_plays'], 
                color='gold', edgecolor='black')
        ax4.set_xlabel('Bi-cluster', fontsize=11)
        ax4.set_ylabel('Total Plays', fontsize=11)
        ax4.set_title('Total Plays per Bi-cluster', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        title: str = "Model Comparison"
    ) -> plt.Figure:
        """
        Plot model comparison metrics
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison dataframe
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.Figure
        """
        n_cols = max(len(comparison_df.columns), 1) if comparison_df is not None else 1
        n_rows = max(len(comparison_df), 1) if comparison_df is not None else 1
        fig_width = max(12, min(24, 1.9 * n_cols))
        fig_height = max(4, min(12, 1.1 * n_rows + 2.8))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if comparison_df is None or comparison_df.empty:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                'No comparison data available.',
                ha='center',
                va='center',
                fontsize=12,
                transform=ax.transAxes
            )
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            return fig
        
        # Create table
        ax.axis('tight')
        ax.axis('off')

        n_cols = max(len(comparison_df.columns), 1)
        if n_cols == 1:
            col_widths = [0.9]
        else:
            first_col = 0.18
            remaining_width = 0.96 - first_col
            each = max(remaining_width / (n_cols - 1), 0.07)
            col_widths = [first_col] + [each] * (n_cols - 1)
        
        table = ax.table(
            cellText=comparison_df.values,
            colLabels=comparison_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=col_widths
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9 if n_cols > 8 else 10)
        table.scale(1, 1.6)
        
        # Style header
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(comparison_df) + 1):
            for j in range(len(comparison_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
