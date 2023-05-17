import numpy as np
import pandas as pd

class Distance:
    def __init__(self, data, simu, covariates):
        """Init by check input types and prepare dataframe.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
        """
        self.checktype(data)
        self.checktype(simu)

        self.covariates = covariates 
        self.df = self.prepare(data, simu)

    def checktype(self, data):
        """Check if the input data is pandas dataframe.

        Args:
            data (dataframe): Data or simulation to check type.

        Raises:
            TypeError: Only numpy arrays and pandas dataframe are supported for the input data/simulation type!
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data=data)
        elif isinstance(data, pd.   core.frame.DataFrame):
            pass
        else:
            raise TypeError('Only numpy arrays and pandas dataframe are supported for the input data/simulation type!')

    def prepare(self, data, simu):
        """Prepare data and simulation to be a combined dataframe which only includes the covariates as columns.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.

        Returns:
            combined (dataframe): Combined dataframe which only includes the covariates as columns.
        """
        # Pick out the interested columns
        data_refined =  data[self.covariates]
        simu_refined =  simu[self.covariates]

        # Add labels
        data_refined['is_simu'] = False
        simu_refined['is_simu'] = True
        
        # Concatenate simulation and data
        combined = pd.concat([simu_refined, data_refined], ignore_index=True)
        combined.reset_index(inplace = True)

        return combined


class Mahalanobis(Distance):
    """Distance computation based on Mahalnobis distance, which is normalized Euclidean distances.
    """
    def __init__(self, data, simu, covariates):
        """Init by computing inverse sigma matrix.

        Args:
            data (dataframe): Data to select.
            simu (dataframe): Simulation to refer.
            covariates (list): A list of strings, corresponding to field names of covariates in matching.
        """
        super().__init__(data, simu, covariates)
        self.method = 'Mahalanobis'
        self.sig_i = self.calc_inverse_sig()

    def calc_inverse_sig(self):
        """Compute the inverse of covariance matrix.

        Returns:
            sig_i (2darray): The inverse of covariance matrix.
        """
        # Get the covariates into an array
        df = self.df
        x = np.array([df[covariate] for covariate in df.columns[1:-1]])

        # Compute inverse of covariance of covariates matrix
        sig = np.cov(x)
        # In case we only have one covariate
        if len(np.shape(sig))==0:
            sig = np.array([[sig]])

        sig_i = np.linalg.inv(sig)
        
        return sig_i

    def prep_matrix(self):
        """Extract covariate matrix from data and simu.

        Returns:
            data_matrix (2darray): Data covariate matrix, where each row is a feature vector of certain event.
            simu_matrix (2darray): Simu covariate matrix, where each row is a feature vector of certain event.
        """
        df = self.df
        data_matrix = df[~df['is_simu']].drop(columns=['index', 'is_simu']).to_numpy()
        simu_matrix = df[df['is_simu']].drop(columns=['index', 'is_simu']).to_numpy()
        return data_matrix, simu_matrix

    def calc_distance_matrix(self):
        """Compute the distance between two events (one simulation one data).

        Returns:
            distance_matrix (2darray): The matrix {i,j} distance between two events i and j.
        """
        data_matrix, simu_matrix = self.prep_matrix()

        diff = simu_matrix[:, np.newaxis] - data_matrix
        distance_matrix = np.einsum('ijk,kl,ijl->ij', diff, self.sig_i, diff)
        distance_matrix = np.array(distance_matrix)

        return distance_matrix

    def calc_distances(self):
        """Get the dataframe of distance between two events.

        Returns:
            distances (dataframe): Distance between two events.
        """
        df = self.df
        distance_matrix = self.calc_distance_matrix()

        simu_index = df.loc[df['is_simu'], 'index'].to_numpy()
        data_index = df.loc[~df['is_simu'], 'index'].to_numpy()
        simu_index, data_index = np.meshgrid(simu_index, data_index, indexing='ij')

        distances = pd.DataFrame({
            'simu_index': simu_index.ravel(),
            'data_index': data_index.ravel(),
            'distance': distance_matrix.ravel()
        })

        return distances
