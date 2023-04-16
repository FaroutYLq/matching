from .distance import *

class Match:
    def __init__(self, data, simu, covariates, distance='mahalanobis'):
        """Init matching class by computing distances with definition of your choice.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
        """
        distance = distance.capitalize()
        self.distance = distance
        distance_class = eval(distance)

        print('Computing %s distances...'%(distance)) 
        distances = distance_class(data, simu, covariates).calc_distances()
        print('Distances have been computed')

        self.distances = distances


class NearestNeighbor(Match):
    """Each simulation event is matched to a data event with shortest distance to it.
    """
    def __init__(self, data, simu, covariates, distance='mahalanobis'):
        super().__init__(data, simu, covariates, distance)
        self.method = 'NearestNeighbor'

    def find_matches(self):
        """Find matches for each simulation event. Each simulation event is matched to a data event
        with shortest distance to it.

        Returns:
            df_matched (dataframe): matched result in this format "simu_index - data_index - distance"
        """
        print('Computing %s matching...'%(self.method))
        df = self.distances

        s_argmin = df.groupby(['simu_index'])['distance'].idxmin()
        df_matched = df.loc[s_argmin][['simu_index', 'data_index', 'distance']]

        print('Matching has been computed.')

        return df_matched
    