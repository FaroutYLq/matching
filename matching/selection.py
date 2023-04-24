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
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(simu, np.ndarray):
            data = pd.DataFrame(simu)
            
        self.distance = distance
        self.match = match
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
        selected_data_idx = np.array(selected_counts.index)
        selected_data_mask = np.zeros(self.data_counts, dtype=np.bool)
        selected_data_mask[selected_data_idx] = True

        # Print how much data survived selection
        if _print_survival_ratio:
            print('Survival ratio:', selected_data_mask.sum()/self.data_counts)

        return selected_data_mask
