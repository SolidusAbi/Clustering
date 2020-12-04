from IPDL import MatrixBasedRenyisEntropy
from .MatrixEstimator import MatrixEstimator
from torch import nn
from utils import moving_average as mva

class InformationPlane():
    '''
        Pass a list of tensor which contents the matrices in order to calculate the
        MutualInformation
    '''

    def __init__(self, model: nn.Module):
        self.matrices_per_layers = []

        test = MatrixEstimator(gamma=0.8)
        
        # First element corresponds to input A matrix and last element
        # is the output A matrix
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrices_per_layers.append(module)

        if len(self.matrices_per_layers) < 3:
            raise Exception('There are not enough MatrixEstimators')

        self.Ixt = []
        self.Ity = []
        for i in range(len(self.matrices_per_layers)-2):
            self.Ixt.append([])
            self.Ity.append([])
    
    def calculate_mi(self):
        Ax = self.matrices_per_layers[0].A
        Ay = self.matrices_per_layers[-1].A

        for idx, matrix_estimator in enumerate(self.matrices_per_layers[1:-1]):
            self.Ixt[idx].append(MatrixBasedRenyisEntropy.mutualInformation(Ax, matrix_estimator.A).cpu())
            self.Ity[idx].append(MatrixBasedRenyisEntropy.mutualInformation(matrix_estimator.A, Ay).cpu())

    def get_mi(self, moving_average_n = 0):
        if moving_average_n == 0:
            return self.Ixt, self.Ity
        else:
            filter_Ixt = list(map(lambda Ixt: mva(Ixt, moving_average_n), self.Ixt))
            filter_Ity = list(map(lambda Ity: mva(Ity, moving_average_n), self.Ity))
            return filter_Ixt, filter_Ity

    def get_input_matrix(self):
        return self.matrices_per_layers[0].A
    
    def get_output_matrix(self):
        return self.matrices_per_layers[-1].A