"""
Configuration file for Last.fm Clustering Project
"""

import os

# Paths - Handle both script and interactive (notebook) execution
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # In interactive/notebook environment, __file__ may not be defined
    BASE_DIR = os.path.abspath(os.getcwd())
    if BASE_DIR.endswith('notebooks'):
        BASE_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Data files
LISTENING_FILE = os.path.join(DATA_DIR, 'userid-timestamp-artid-artname-traid-traname.tsv')
PROFILE_FILE = os.path.join(DATA_DIR, 'userid-profile.tsv')

# Sampling parameters (due to large dataset size)
SAMPLE_USERS = 500  # Number of users to sample for analysis
MIN_ARTIST_PLAYS = 5  # Minimum plays for an artist to be included
MIN_USER_PLAYS = 20  # Minimum plays for a user to be included

# Clustering parameters
N_CLUSTERS_RANGE = range(3, 11)  # Range of clusters to try
OPTIMAL_CLUSTERS = 5  # Default optimal number of clusters

# Bi-clustering parameters
N_BICLUSTERS = 4  # Number of bi-clusters (user-artist groups)

# Visualization
RANDOM_STATE = 42
FIGSIZE = (12, 8)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
