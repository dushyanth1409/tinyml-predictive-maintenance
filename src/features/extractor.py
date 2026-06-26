"""
Feature Extractor
------------------
Combines time + frequency domain features into a single feature vector.
Processes raw signals through the full pipeline:

  raw signal → normalize → window → extract features → feature matrix

Output shape: (N_windows, N_features) where N_features = 22
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.features.time_domain import extract_time_features
from src.features.frequency_domain import extract_frequency_features


class FeatureExtractor:
    def __init__(self,
                 sample_rate:  float = 12_000.0,
                 window_size:  int   = 1024,
                 hop_size:     int   = 512,
                 norm_method:  str   = "zscore"):
        self.sample_rate = sample_rate
        self.pipeline = PreprocessingPipeline(
            norm_method=norm_method,
            window_size=window_size,
            hop_size=hop_size,
        )

    def extract_window(self, window: np.ndarray) -> dict:
        """Extract all features from a single window."""
        time_feats = extract_time_features(window)
        freq_feats = extract_frequency_features(window, self.sample_rate)
        return {**time_feats, **freq_feats}

    def extract_signal(self, signal: np.ndarray) -> pd.DataFrame:
        """
        Extract features from all windows of a single signal.

        Returns
        -------
        pd.DataFrame  shape (N_windows, N_features)
        """
        windows  = self.pipeline.process(signal)
        features = [self.extract_window(w) for w in windows]
        return pd.DataFrame(features)

    def build_dataset(self, signals: list, labels: list,
                      agg: str = "mean") -> tuple[pd.DataFrame, np.ndarray]:
        """
        Build a feature matrix from a list of signals + labels.

        Parameters
        ----------
        signals : list of np.ndarray
        labels  : list of int
        agg     : str  How to aggregate window features per signal.
                       'mean' | 'max' | 'first'

        Returns
        -------
        X : pd.DataFrame   (N_signals, N_features)
        y : np.ndarray     (N_signals,) integer labels
        """
        rows = []
        for sig in signals:
            df = self.extract_signal(sig)
            if agg == "mean":
                rows.append(df.mean())
            elif agg == "max":
                rows.append(df.max())
            elif agg == "first":
                rows.append(df.iloc[0])
            else:
                raise ValueError(f"Unknown aggregation: {agg}")

        X = pd.DataFrame(rows).reset_index(drop=True)
        y = np.array(labels)
        return X, y

    @property
    def feature_names(self) -> list:
        """Return ordered list of feature names."""
        dummy = np.random.randn(self.pipeline.window_size)
        return list(self.extract_window(dummy).keys())
