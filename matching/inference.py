import numpy as np
from .match import *
from tqdm import tqdm
import pandas as pd

# Should be enough for 20GB RAM
DATA_BATCH_SIZE = 2500
SIMU_BATCH_SIZE = 15000

class Inference:
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor', central_fraction=0.98):
        """Init Causal Inference by getting matching result.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
            match (str, optional): Matching method definition. Defaults to 'NearestNeighbor'.
            central_fraction (float): Central fraction to align covariates in data and simu. Default to be 0.98.
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(simu, np.ndarray):
            simu = pd.DataFrame(simu)

        self.distance = distance
        self.match = match

        # Align range of covariates to avoid artifacts
        self.central_fraction = central_fraction
        data_mask, simu_mask = self.align_covariate_range(data, simu, covariates)
        data = data[data_mask]
        simu = simu[simu_mask]

        self.data = data
        self.simu = simu
        self.data_counts = len(data)
        self.simu_counts = len(simu)
        self.data_batch_size = DATA_BATCH_SIZE
        self.simu_batch_size = SIMU_BATCH_SIZE
        
        # Get segmented data batches in the form of a dictionary
        optimized_batch_size = self.get_batch_sizes(data, DATA_BATCH_SIZE)
        self.data_batch_size = optimized_batch_size
        data_batches = self.get_batches(data, batch_size=DATA_BATCH_SIZE)

        # Get match class to alleiviate computational burden.
        batch_num = int(len(data)/DATA_BATCH_SIZE) + 1
        match_class = eval(match)
        for i in tqdm(range(batch_num)):
            if len(simu) <= SIMU_BATCH_SIZE:
                simu_batches = simu
            else:
                if len(simu) <= SIMU_BATCH_SIZE:
                    simu_batches = simu
                else:
                    simu_batches = simu.sample(n=SIMU_BATCH_SIZE)
            match_object = match_class(data_batches[i], simu_batches, covariates, distance)
            matches_i = match_object.find_matches()

            # Decode data index
            matches_i['data_index'] += i*optimized_batch_size
            matches_i['data_index'] -= min(len(simu), SIMU_BATCH_SIZE)

            if i == 0:
                matches = matches_i
            else:
                matches = pd.concat([matches, matches_i])
        self.matches = matches


    def align_covariate_range(self, data, simu, covariates):
        """Align range of covariates to avoid artifacts.
        Returns such a mask for "data" and "simu" respectively: For each element "cov" in "covariates", 
        that the mask marks the intersection of central specified percentile of the distribution of 
        data[cov] and simu[cov].

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.

        Returns:
            data_mask (ndarray, bool):  
        """
        central_fraction = self.central_fraction

        assert central_fraction>0 and central_fraction<1, 'central_fraction must be between 0 and 1!'
        percent = (1-central_fraction)/2

        data_mask = pd.Series([True] * len(data), index=data.index)
        simu_mask = pd.Series([True] * len(simu), index=simu.index)

        for cov in covariates:
            data_left_percentile = data[cov].quantile(percent)
            data_right_percentile = data[cov].quantile(1-percent)
            simu_left_percentile = simu[cov].quantile(percent)
            simu_right_percentile = simu[cov].quantile(1-percent)

            intersection_low = max(data_left_percentile, simu_left_percentile)
            intersection_high = min(data_right_percentile, simu_right_percentile)

            data_mask &= (data[cov] >= intersection_low) & (data[cov] <= intersection_high)
            simu_mask &= (simu[cov] >= intersection_low) & (simu[cov] <= intersection_high)

        return data_mask, simu_mask
    

    def get_batch_sizes(self, events, batch_size):
        """Optimize the batch sizes based on the rough batch size input.

        Args:
            events (dataframe): Either data or simu.
            batch_size (int): Number of events per batch

        Returns:
            optimized_batch_size (int): Optimized batch size.
        """
        batch_num = int(len(events)/batch_size) + 1
        optimized_batch_size = int(len(events)/batch_num)
        return optimized_batch_size

    def get_batches(self, events, batch_size):
        """Segment events into batches as a dictionary.

        Args:
            events (dataframe): Either data or simu.
            batch_size (int): Number of events per batch

        Returns:
            event_batches (dataframe): Events segmented into batches as a dictionary.
        """
        optimized_batch_size = self.get_batch_sizes(events, batch_size)
        batch_num = int(len(events)/optimized_batch_size)
        event_batches = {}
        for i in range(batch_num):
            event_batches[i] = events[i * optimized_batch_size : (i+1) * optimized_batch_size]
        return event_batches

    def match_simu(self):
        """Match one data event to every simulation event.

        Returns:
            matched_data (dataframe): Data matched to simulation
        """
        matches = self.matches

        # Count how many time one data event has been matched
        matched_counts = matches['data_index'].value_counts()
        matched_counts = matched_counts.to_frame(name='counts')

        # Assumed simulation always has smaller indices
        selected_data_idx = np.repeat(matched_counts.index, matched_counts['counts'].values)

        # matched data
        matched_data = self.data.iloc[selected_data_idx]

        return matched_data
