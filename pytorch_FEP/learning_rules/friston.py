import torch

from .learning_rule import LearningRule


class FristonsRule(LearningRule):
    """Friston Free Energy learning rule fast implementation.

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k: Ranking parameter
    """

    def __init__(self, precision=1e-32, norm=2, normalize=False, state_prior=1):
        super().__init__()
        self.precision = precision
        self.normalize = normalize
        self.norm = norm
        self.state_prior = state_prior

    def init_layers(self, layers: list):
        for layer in [lyr.layer for lyr in layers]:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
                layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        # inputs: [batch_size, input_size] [batch_size, M]
        # weights: [num_hid, input_size] [N, M]
        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0] # N, num of hidden states
        input_size = inputs[0].shape[0] # M, num of observations

        # TODO: WIP
        if self.normalize:
            norm = torch.norm(inputs, dim=1)
            norm[norm == 0] = 1
            inputs = torch.div(inputs, norm.view(-1, 1))

        # transpose: weights: [input_size, num_hid] [M, N]
        weights = torch.t(weights)

        # Calculate 1-inputs & 1-weights
        inputs_hat = 1 - inputs
        weights_hat = 1 - weights

        # state prior initiation
        # D = torch.normal(mean=self.state_prior, std=0.05, size=(1, num_hidden_units))
        # lnD = torch.log(D)

        # qA = torch.div(weights, weights + weights_hat)
        # qA_hat = 1 - qA

        log_sum = torch.log(torch.max(torch.tensor(1e-6), weights + weights_hat))
        qlnA = torch.log(torch.max(torch.tensor(1e-6), weights)) - log_sum
        qlnA_hat = torch.log(torch.max(torch.tensor(1e-6), weights_hat)) - log_sum

        # Inference: get the medium result 
        # qs: [batch_size, N]
        qs = torch.exp(torch.matmul(inputs, qlnA) + torch.matmul(inputs_hat, qlnA_hat))# + lnD)
        #qs_hat = 1 - qs

        # Learning: the update rule for weights
        ds = torch.matmul(inputs.t(), qs)

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.true_divide(ds, nc)

        # Normalize over each hidden unit
        d_w = torch.nn.functional.normalize(d_w, p=2, dim=0)

        return d_w
