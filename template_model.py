"""
Model_to_federate is an abstract class defining the template that every ML model used in the FL simulation must follow.

"""


from abc import ABC, abstractmethod


class Model_to_federate(ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def train(self, net, X_train, Y_train, parameters_training, parameters_model = None):
        pass

    @abstractmethod
    def test(self, net, X_test, Y_test, parameters_training, parameters_model = None):
        pass

    @abstractmethod
    def train_pooled_data (self, net, X_train, Y_train, X_val, Y_val, parameters_training, parameters_model = None):
        pass

    @abstractmethod
    def parameter_aggregation_fn(self, net, parameters):
        pass

    @abstractmethod
    def get_parameters(self, net):
        pass

    @abstractmethod
    def set_parameters(self, parameters, net):
        pass
