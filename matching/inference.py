import numpy as np
from .match import *

class Inference:
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        """Init Causal Inference by getting matching result.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
            match (str, optional): Matching method definition. Defaults to 'NearestNeighbor'.
        """
        self.distance = distance
        self.match = match
        self.data = data
        self.data_counts = len(data)
        self.simu_counts = len(simu)
        
        match_class = eval(match)
        match_object = match_class(data, simu, covariates, distance)

        # Load matches
        self.matches = match_object.find_matches()

    def match_simu(self):
        """Match one data event to every simulation event.

        Returns:
            matched_data (dataframe): Matched data events.
        """
        matches = self.matches

        # Count how many time one data event has been matched
        matched_counts = matches['data_index'].value_counts()
        matched_counts = matched_counts.to_frame(name='counts')

        # Assumed simulation always has smaller indices
        selected_data_idx = np.repeat(matched_counts.index, matched_counts['counts'].values)
        selected_data_idx -=  self.simu_counts 

        # matched data
        matched_data = self.data[selected_data_idx]

        return matched_data
