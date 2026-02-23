"""
Utility functions for the Last.fm Clustering Project
"""

import os
import json
import pickle
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk
    
    Parameters:
    -----------
    model : Any
        Model object to save
    filepath : str
        Path to save the model
    """
    logger.info(f"Saving model to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Model saved successfully.")


def load_model(filepath: str) -> Any:
    """
    Load model from disk
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    Any
        Loaded model object
    """
    logger.info(f"Loading model from {filepath}...")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("Model loaded successfully.")
    return model


def save_results(results: dict, filepath: str) -> None:
    """
    Save results to JSON file
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filepath : str
        Path to save results
    """
    logger.info(f"Saving results to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        import datetime as dt
        import numpy as np
        import pandas as pd

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, pd.Timestamp, dt.datetime, dt.date)):
            return str(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_types(item) for item in sorted(obj, key=str)]
        return obj
    
    results_converted = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    logger.info("Results saved successfully.")


def create_project_structure(base_dir: str) -> None:
    """
    Create project directory structure
    
    Parameters:
    -----------
    base_dir : str
        Base directory for the project
    """
    directories = [
        'output',
        'output/models',
        'output/plots',
        'output/results'
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
