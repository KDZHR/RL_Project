
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv2d_size_out(size, kernel_size, stride):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


class DuelingNetwork(nn.Module):
    """
    Implement the Dueling DQN logic.
    """
    def __init__(self, n_actions, inp_size, hidden_size) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.inp_size = inp_size
        self.hidden_size = hidden_size

        self.fc_value = nn.Linear(self.inp_size, self.hidden_size)
        self.relu_value = nn.ReLU()

        self.fc_advantage = nn.Linear(self.inp_size, self.hidden_size)
        self.relu_advantage = nn.ReLU()

        self.value_stream = nn.Linear(self.hidden_size, 1)
        self.advantage_stream = nn.Linear(self.hidden_size, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, x.shape  # (batch_size, n_features)
        # When calculating the mean advantage, please, remember, x is a batched input!

        value = self.fc_value(x)
        value = self.relu_value(value)
        value = self.value_stream(value)

        advantage = self.fc_advantage(x)
        advantage = self.relu_advantage(advantage)
        advantage = self.advantage_stream(advantage)

        return value + advantage - advantage.mean(dim=1, keepdim=True)
        

class GradScalerFunctional(torch.autograd.Function):
    """
    A torch.autograd.Function works as Identity on forward pass
    and scales the gradient by scale_factor on backward pass.
    """
    @staticmethod
    def forward(ctx, input, scale_factor):
        ctx.scale_factor = scale_factor
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale_factor = ctx.scale_factor
        grad_input = grad_output * scale_factor
        return grad_input, None


class GradScaler(nn.Module):
    """
    An nn.Module incapsulating GradScalerFunctional
    """
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return GradScalerFunctional.apply(x, self.scale_factor)


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.scale_factor = 1.0 / np.sqrt(2.0)

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful

        num_channels, cur_layer_img_w, cur_layer_img_h = self.state_shape

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size=3, stride=2)
        cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size=3, stride=2)
        cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size=3, stride=2)
        cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.scaler = GradScaler(scale_factor=self.scale_factor)
        self.dueling = DuelingNetwork(n_actions, cur_layer_img_w * cur_layer_img_h * 64, 256)


    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        x = self.conv1(state_t)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.scaler(x)

        x = self.dueling(x)

        return x


    @property
    def device(self):
        return next(self.parameters()).device


    @torch.inference_mode()
    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        """
        like forward, but works on numpy arrays, not tensors
        """
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        qvalues = self(states)
        return qvalues.cpu().detach().numpy()


    def sample_actions_by_qvalues(self, qvalues: np.ndarray, greedy: bool = False) -> np.ndarray:
        """pick actions given qvalues based on epsilon-greedy exploration strategy."""
        batch_size, n_actions = qvalues.shape
        eps = self.epsilon
        
        if not greedy and np.random.rand() < eps:
            return np.random.choice(n_actions, size=batch_size)
        else:
            return np.argmax(qvalues, axis=1)


    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        qvalues = self.get_qvalues(states)
        return self.sample_actions_by_qvalues(qvalues, greedy)
