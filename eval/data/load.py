import pandas as pd
import numpy as np
import os


def load_test_df():
    """Load the main test dataset"""
    file_path = os.path.join(os.path.dirname(__file__), "test_data.csv.gz")
    return pd.read_csv(file_path, compression='gzip')

def load_author_styles():
    """Load vector representations for main target styles"""
    file_path = os.path.join(os.path.dirname(__file__), "author_styles.npz")
    with np.load(file_path) as loaded:
        author_styles = {key: loaded[key] for key in loaded}
    return author_styles

def load_llm_data():
    """Load TST results performed by LLMs (manually)."""
    file_path = os.path.join(
        os.path.dirname(__file__), "llms_with_scores.csv.gz"
    )
    return pd.read_csv(file_path, compression='gzip')