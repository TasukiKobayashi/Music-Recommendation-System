# Last.fm Music Clustering Project

A comprehensive machine learning project comparing **Clustering** vs **Bi-Clustering** approaches for music recommendation using the Last.fm dataset.

## 📋 Project Overview

This project implements and compares two clustering approaches:

1. **Traditional Clustering (K-Means)**: Groups users based on their listening patterns
2. **Bi-Clustering (Spectral Co-Clustering)**: Simultaneously clusters users AND artists to find coherent user-artist groups

## 🎯 Objectives

- Load and preprocess the Last.fm-1K dataset (19M+ listening records, 992 users, 177K artists)
- Create meaningful features from user listening behavior
- Implement K-Means clustering for user segmentation
- Implement bi-clustering to discover user-artist communities
- Comprehensively compare both approaches using multiple metrics
- Visualize and interpret the results

## 📁 Project Structure

```
lastfm-clustering-project/
│
├── config.py                 # Configuration and parameters
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── clustering_model.py  # K-Means clustering implementation
│   ├── biclustering_model.py   # Bi-clustering implementation
│   ├── evaluation.py        # Model evaluation and comparison
│   ├── visualization.py     # Plotting and visualization
│   └── utils.py             # Utility functions
│
├── notebooks/
│   └── LastFM_Clustering_Analysis.ipynb  # Main analysis notebook
│
└── output/                  # Generated outputs
    ├── models/              # Saved models
    ├── plots/               # Generated plots
    └── results/             # Result files
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone or navigate to the project directory:
```bash
cd lastfm-clustering-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the Last.fm dataset is in the project directory:
```
dataset/
    ├── userid-profile.tsv
    └── userid-timestamp-artid-artname-traid-traname.tsv
```

### Usage

Run the main Jupyter notebook:
```bash
jupyter notebook notebooks/LastFM_Clustering_Analysis.ipynb
```

Run the full pipeline from terminal (no notebook):
```bash
python main.py
```

Run the simple Flask demo (after generating outputs with `main.py`):
```bash
python demo/app.py
```

Then open:
```text
http://127.0.0.1:5000
```

Optional arguments:
```bash
python main.py --nrows 800000 --sample-users 400 --k 6 --biclusters 4
python main.py --skip-optimal-k-search
```

## 📊 Dataset Information

**Last.fm-1K Dataset**
- **Total Records**: 19,150,868 listening events
- **Users**: 992 unique users
- **Artists**: 177,000+ unique artists
- **Time Period**: User listening history until May 5, 2009
- **User Profiles**: Gender, age, country, registration date

## 🔬 Methodology

### 1. Data Preprocessing
- Load listening history and user profiles
- Sample users for computational efficiency
- Filter low-frequency users and artists
- Handle missing values

### 2. Feature Engineering
- Create user-artist play count matrix
- Generate user-level features:
  - Total plays, unique artists, unique tracks
  - Artist/track diversity metrics
  - Temporal patterns
  - Top artist concentration
- Normalize and scale features

### 3. Clustering (K-Means)
- Find optimal K using elbow method and silhouette score
- Fit K-Means on user features
- Analyze cluster characteristics
- Reduce dimensions (PCA) for visualization

### 4. Bi-Clustering (Spectral Co-Clustering)
- Apply to user-artist matrix
- Discover user-artist communities
- Analyze bi-cluster density and coverage
- Identify characteristic artists per bi-cluster

### 5. Evaluation & Comparison
- **Clustering Metrics**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- **Bi-Clustering Metrics**: Density, coverage, cluster balance
- **Agreement Metrics**: Adjusted Rand Index, Normalized Mutual Information
- Visualizations and interpretations

## 📈 Key Evaluation Metrics

### Clustering (K-Means)
- **Silhouette Score**: Measures how similar users are within clusters (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

### Bi-Clustering
- **Bi-cluster Density**: Proportion of non-zero entries in user-artist sub-matrices
- **Coverage**: Percentage of users/artists assigned to bi-clusters
- **Cluster Balance**: Distribution uniformity across clusters

### Comparison
- **Adjusted Rand Index**: Agreement between clustering assignments
- **Normalized Mutual Information**: Shared information between clusterings

## 🎨 Visualizations

The project generates multiple visualizations:
- Elbow curves for optimal K selection
- Silhouette score plots
- 2D cluster visualizations (PCA projection)
- Cluster size distributions
- Feature importance heatmaps
- Bi-cluster heatmaps
- Bi-cluster statistics
- Model comparison tables

## 🔑 Key Findings

Results will show:
- **Clustering** groups users by overall listening behavior profiles
- **Bi-clustering** identifies specific user-artist communities
- Different perspectives on user segmentation for recommendations
- Trade-offs between granularity and interpretability

## 📝 License

This project uses the Last.fm-1K dataset, which is distributed with permission from Last.fm for non-commercial use.

## 🙏 Acknowledgments

- Last.fm for providing the dataset
- Oscar Celma for collecting the data
- Inspired by music recommendation research

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Note**: This is an educational project demonstrating clustering techniques for music recommendation.
