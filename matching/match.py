from .distance import *

class Match:
    def __init__(self, data, simu, covariates, distance='Mahalanobis'):
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
        distance_object = distance_class(data, simu, covariates)

        #print('Computing %s distances...'%(distance)) 
        distances = distance_object.calc_distances()
        #print('Distances have been computed')
        self.df = distance_object.df

        self.distances = distances


class NearestNeighbor(Match):
    """Each simulation event is matched to a data event with shortest distance to it.
    """
    def __init__(self, data, simu, covariates, distance='Mahalanobis'):
        """Init by doing nothing new.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
            distance (str, optional): Distance definition. Defaults to 'mahalanobis'.
        """
        super().__init__(data, simu, covariates, distance)
        self.method = 'NearestNeighbor'

    def find_matches(self):
        """Find matches for each simulation event. Each simulation event is matched to a data event
        with shortest distance to it.

        Returns:
            df_matched (dataframe): matched result in this format "simu_index - data_index - distance"
        """
        #print('Computing %s matching...'%(self.method))
        df = self.distances

        s_argmin = df.groupby(['simu_index'])['distance'].idxmin()
        matches = df.loc[s_argmin][['simu_index', 'data_index', 'distance']]

        #print('Matching has been computed.')

        return matches
    