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
        if isinstance(data, np.array):
            data = pd.DataFrame(data=data)
        elif isinstance(data, pd.core.frame.DataFrame):
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

def Mahalanobis(Distance):
    def __init__(self, data, simu, covariates):
        super().__init__(data, simu, covariates)
        self.method = 'mahalanobis'
        self.sig_i = self.calc_inverse_sig()

    def calc_inverse_sig(self):
        # Get the covariates into an array
        df = self.df
        x = np.array([df[covariate] for covariate in df.columns[1:]])

        # Compute inverse of covariance of covariates matrix
        sig = np.cov(x)
        sig_i = np.linalg.inv(sig)
        
        return sig_i

    def prep_matrix(self):
        df = self.df
        data_matrix = df[~df['is_simu']].drop(columns=['index', 'is_simu']).to_numpy()
        simu_matrix = df[df['is_simu']].drop(columns=['index', 'is_simu']).to_numpy()
        return data_matrix, simu_matrix

    def calc_distance_matrix(self):
        data_matrix, simu_matrix = self.prep_matrix()

        diff = simu_matrix[:, np.newaxis] - data_matrix
        distance_matrix = np.einsum('ijk,kl,ijl->ij', diff, sig_i, diff)
        distance_matrix = np.array(distance_matrix)

        return distance_matrix

    def calc_all_distance(self):
        df = self.df
        distance_matrix = self.calc_distance_matrix()

        simu_index = df.loc[df['is_simu'], 'index'].to_numpy()
        data_index = df.loc[~df['is_simu'], 'index'].to_numpy()
        simu_index, data_index = np.meshgrid(simu_index, data_index, indexing='ij')

        distances = pd.DataFrame({
            'simu_index': T_event.ravel(),
            'data_index': C_event.ravel(),
            'distance': distance_matrix.ravel()
        })

        return distances
