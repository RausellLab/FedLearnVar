from template_model import Model_to_federate
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from utils import Metrics
import numpy as np
import flwr as fl
import torch.nn.functional as F
import torch
import copy


"""
Implementation of MLP in a FL setting
"""
class MLP_to_federate(Model_to_federate):

    def __init__(self) -> None:
        super().__init__()
    
    """
    this method defines the training at client side
    - net: the ML model sent by the server (in this case we assume it is a MLP, 
    but other NN architectures can be considered as well)
    - X_train: the samples for training
    - Y_train: the labels of X_train
    - parameters_training: a dictionary containing parameters related to the training phase
    - parameters_model: a dictionary containing additional parameters related to the model
    """
    def train(self, net, X_train, Y_train, parameters_training, parameters_model= None):

        batch_size = parameters_training["batch_size"]
        learning_rate = parameters_training["learning_rate"]
        weight_decay = parameters_training["weight_decay"]
        epochs = parameters_training["epochs"]
        batch_to_iterate = parameters_training["batch_to_iterate"]
        device = torch.device(parameters_training["device"])

        mu = parameters_training["mu"] #this parameter is related to proximal term strategy. The default value is 0.

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = False)

        optimizer = torch.optim.SGD(params = net.parameters(), lr = learning_rate, weight_decay = weight_decay,
                                    momentum = 0.9)
        global_model = copy.deepcopy(net)

        net.train()
        for epoch in range(epochs):

            avg_loss = []
            for i, data in enumerate(trainloader,0):
                
                genetic_variant_data, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                output = net(genetic_variant_data)

                
                proximal_term = 0.0
                for w, w_t in zip(net.parameters(), global_model.parameters()):
                    proximal_term+=(w - w_t).norm(2)

                #This is the update taking into consideration the proximal term mu
                loss = torch.nn.functional.nll_loss(output,labels.long()) + (mu/2)*proximal_term
                loss.backward()
                avg_loss.append(loss.item())
                optimizer.step()

            
            avg_loss = np.average(np.array(avg_loss))
            if (epoch + 1) % 10 == 0: #printing the mean loss every 10 epochs
                print(f"Epoch {epoch} avg loss {avg_loss}")

    """
    This method defines the evaluation at client side
     - net: the ML model sent by the server (in this case we assume it is a MLP, 
    but other NN architectures can be considered as well)
    - X_test: the samples for testing
    - Y_test: the labels of X_test
    - parameters_evaluation: a dictionary containing parameters related to the evaluation phase
    - parameters_model: a dictionary containing additional parameters related to the model

    Returns: the test lost, the lenght of the test set, a dictionary containing the evaluation metric,
    the true labels, and the prediction score for each test sample

    """
    
    def test(self, net, X_test, Y_test, parameters_evaluation, parameters_model=None):

        device = torch.device(parameters_training["device"])
        metrics = Metrics()

        if X_test is None:
            return 0.0 , 0, metrics.get_dict(), None, None # the property metrics.length = O gives the idea that there is no data set to test

        net.eval() 
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size = len(Y_test), shuffle = False)

        metrics.set_length(len(Y_test))

        with torch.no_grad():
            
            for data in test_dataloader: #it is supposed just one test set
                cnv_data, labels = data[0].to(device), data[1].to(device)
                output = net(cnv_data)
                loss_ = torch.nn.functional.nll_loss(output, labels.long())
                prediction2 = output.cpu().detach().numpy()
                prediction_prob = prediction2[:,1]
                _, predicted_labels = torch.max(output.data.cpu(), 1)

                #getting the evaluation metrics
                metrics.set_loss(loss_.item())
                metrics.get_metrics(labels.cpu(), predicted_labels, prediction_prob) #this gives me the flexibility to incorporate more metrics in a future without changing this part of the code

        return metrics.loss, metrics.length, metrics.get_dict(), labels.cpu().numpy(), prediction_prob


    """
    This method defines the function for aggregating the parameters at server side.
    - net: the NN architecture 
    - parameters: parameters aggregated by the server
    - isFedOpt: flag representing if a FedOpt strategy was used (FedAdagrad, FedAdam, FedYogi)
    Returns: the parameters converted to flower weights
    """

    def parameter_aggregation_fn(self, net, parameters, isFedOpt = False):
        return fl.common.parameters_to_weights(parameters)

    """
    This method returns the parameters of the neural network, excluding batch normalization layers if present
    """

    def get_parameters(self, net):
        return [val.cpu().numpy() for name, val in net.state_dict().items() if 'bn' not in name]

    """
    This method update the parameters of the neural network
    - parameters: the parameters aggregated by the server
    - net: the NN architecture
    """
    
    def set_parameters(self, parameters, net):
        #set model parameters from a list of NumPy ndarrays
        keys = [k for k in net.state_dict().keys() if 'bn' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict = False)
