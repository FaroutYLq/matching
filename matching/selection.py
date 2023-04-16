import numpy as np
from .match import *

class Selection:
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        self.distance = distance
        self.match = match
        
        match_class = eval(match)
        match_object = match_class(data, simu, covariates, distance)

        # Load matches
        self.matches = match_object.find_matches()

class MinimumResamplingRate(Selection):
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        super().__init__(data, simu, covariates, distance, match)
        self.data = data
        self.data_counts = len(data)
        self.simu_counts = len(simu)

    def select(self, min_resample_rate):
        matches = self.matches

        # Count how many time one data event has been matched
        matched_counts = matches['data_index'].value_counts()
        matched_counts = matched_counts.to_frame(name='counts')

        # Required minimum sampling rate
        selected_mask = matched_counts['counts']  >= min_resample_rate*self.simu_counts/self.data_counts
        selected_counts = matched_counts[selected_mask]

        # Assumed simulation always has smaller indices
        selected_data_idx = np.array(selected_counts.index) - self.simu_counts
        selected_data_mask = np.zeros(self.data_counts, dtype=np.bool)
        selected_data_mask[selected_data_idx] = True

        return selected_data_mask
    
        

