import numpy as np
from .match import *
from tqdm import tqdm

# Should be enough for 20GB RAM
DATA_BATCH_SIZE = 2500
SIMU_BATCH_SIZE = 15000

class Selection:
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        """Init by getting matching result. If input data or simu is big, they will be segmented into batches.
        For data, they will be loaded in sequence, but for simulation they will be assiged randomly.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
            match (str, optional): Matching method definition. Defaults to 'NearestNeighbor'.
        """
        self.distance = distance
        self.match = match

        # Get segmented data batches in the form of a dictionary
        data_batches = self.get_batches(data, batch_size=DATA_BATCH_SIZE)

        # Get match class to alleiviate computational burden.
        batch_num = int(len(data)/DATA_BATCH_SIZE)
        match_class = eval(match)
        if len(simu) <= SIMU_BATCH_SIZE:
            for i in tqdm(range(batch_num)):
                match_object = match_class(data_batches[i], simu, covariates, distance)
                matches_i = match_object.find_matches()
                if i == 0:
                    self.matches = matches_i
                else:
                    self.matches = pd.concat([self.matches, matches_i])
        else:
            for i in tqdm(range(batch_num)):
                match_object = match_class(data_batches[i], simu.sample(n=SIMU_BATCH_SIZE), 
                                               covariates, distance)
                matches_i = match_object.find_matches()
                matches_i = match_object.find_matches()
                if i == 0:
                    self.matches = matches_i
                else:
                    self.matches = pd.concat([self.matches, matches_i])

    def get_batch_sizes(self, events, batch_size):
        batch_num = int(len(events)/batch_size)
        optimized_batch_size = int(len(events)/batch_num)
        return optimized_batch_size

    def get_batches(self, events, batch_size):
        optimized_batch_size = self.get_batch_sizes(events, batch_size)
        batch_num = int(len(events)/optimized_batch_size)
        event_batches = {}
        for i in range(batch_num):
            event_batches[i] = events[i * optimized_batch_size : (i+1) * optimized_batch_size]
        return event_batches


class MinimumMatchingRate(Selection):
    """We will check how often a data event is matched to simulation. 
    On average, one data event will be matched simu_counts/data_counts times, and we call this 
    ratio average matching times.
    And min_match_rate defines the data selection criteria: one data event must be matched more than 
    min_match_rate times average matching times, so that it will be selected.
    """
    def __init__(self, data, simu, covariates, distance='Mahalanobis', match='NearestNeighbor'):
        """Init by doing nothing new.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
            match (str, optional): Matching method definition. Defaults to 'NearestNeighbor'.
        """
        super().__init__(data, simu, covariates, distance, match)
        self.data_counts = len(data)
        self.simu_counts = len(simu)
        self.method = 'MinimumMatchingRate'

    def select(self, min_match_rate, _print_survival_ratio=False):
        """Select data based on minimum sample rate. We will check how often a data event is matched to
        simulation. On average, one data event will be matched simu_counts/data_counts times, and we call this 
        ratio average matching times.
        And min_match_rate defines the data selection criteria: one data event must be matched more than 
        min_match_rate times average matching times, so that it will be selected.

        Args:
            min_match_rate (float): Minimum times of average matching to pass selection.
            _print_survival_ratio (bool, optional): Print how much data survived selection. Defaults to False.

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

        # Print how much data survived selection
        if _print_survival_ratio:
            print('Survival ratio:', selected_data_mask.sum()/self.data_counts)

        return selected_data_mask
