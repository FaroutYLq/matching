import numpy as np
from .match import *
from tqdm import tqdm

# Should be enough for 20GB RAM
DATA_BATCH_SIZE = 2500
SIMU_BATCH_SIZE = 15000

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
        
        # Get segmented data batches in the form of a dictionary
        data_batches = self.get_batches(data, batch_size=DATA_BATCH_SIZE)

        # Get match class to alleiviate computational burden.
        batch_num = int(len(data)/DATA_BATCH_SIZE)
        match_class = eval(match)
        for i in tqdm(range(batch_num)):
            if len(simu) <= SIMU_BATCH_SIZE:
                simu_batches = simu
            else:
                simu_batches = simu.sample(n=SIMU_BATCH_SIZE)
            match_object = match_class(data_batches[i], simu_batches, covariates, distance)
            matches_i = match_object.find_matches()

            # Decode data index
            matches_i['data_index'] += i*DATA_BATCH_SIZE
            matches_i['data_index'] -= max(len(simu), SIMU_BATCH_SIZE)

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
        batch_num = int(len(events)/batch_size)
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
