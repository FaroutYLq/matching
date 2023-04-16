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

class MinimumMatchingRate(Selection):
    """We will check how often a data event is matched to simulation. 
    On average, one data event will be matched simu_counts/data_counts times, and we call this 
    ratio average matching times.
    And min_match_rate defines the data selection criteria: one data event must be matched more than 
    min_match_rate times average matching times, so that it will be selected.
    """
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        super().__init__(data, simu, covariates, distance, match)
        self.data = data
        self.data_counts = len(data)
        self.simu_counts = len(simu)

    def select(self, min_match_rate):
        """Select data based on minimum sample rate. We will check how often a data event is matched to
        simulation. On average, one data event will be matched simu_counts/data_counts times, and we call this 
        ratio average matching times.
        And min_match_rate defines the data selection criteria: one data event must be matched more than 
        min_match_rate times average matching times, so that it will be selected.

        Args:
            min_match_rate (float): Minimum times of average matching to pass selection.

        Returns:
            selected_data_mask (1darray of bool): Data selection mask. True means selected.
        """
        matches = self.matches

        # Count how many time one data event has been matched
        matched_counts = matches['data_index'].value_counts()
        matched_counts = matched_counts.to_frame(name='counts')

        # Required minimum matching rate
        selected_mask = matched_counts['counts']  >= min_match_rate*self.simu_counts/self.data_counts
        selected_counts = matched_counts[selected_mask]

        # Assumed simulation always has smaller indices
        selected_data_idx = np.array(selected_counts.index) - self.simu_counts
        selected_data_mask = np.zeros(self.data_counts, dtype=np.bool)
        selected_data_mask[selected_data_idx] = True

        return selected_data_mask
