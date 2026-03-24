
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

        self.fc_q = nn.Linear(self.inp_size, self.hidden_size)
        self.relu = nn.ReLU()

        self.fc_q_head = nn.Linear(self.hidden_size, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, x.shape  # (batch_size, n_features)
        # When calculating the mean advantage, please, remember, x is a batched input!

        q = self.fc_q(x)
        q = self.relu(q)
        q = self.fc_q_head(q).view(-1, self.n_actions)
        return q

class ClassifierDQN(nn.Module):
    """
    Implement the Classifier DQN logic.
    """
    def __init__(self, n_actions, inp_size, hidden_size, num_bins) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.num_bins = num_bins

        self.fc_q = nn.Linear(self.inp_size, self.hidden_size)
        self.relu = nn.ReLU()

        self.advantage_stream = nn.Linear(self.hidden_size, self.n_actions * self.num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, x.shape  # (batch_size, n_features)

        q = self.fc_q(x)
        q = self.relu(q)
        q = self.advantage_stream(q).view(-1, self.n_actions, self.num_bins)
        return q


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, hidden_size, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

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
        self.dueling = DuelingNetwork(n_actions, cur_layer_img_w * cur_layer_img_h * 64, hidden_size=hidden_size)


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


class ClassifierDQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, min_value: float, max_value: float, num_bins: int, hidden_size, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.min_value = min_value
        self.max_value = max_value 
        self.num_bins = num_bins

        support = torch.linspace(min_value, max_value, num_bins + 1)
        centers = (support[:-1] + support[1:]) / 2
        self.register_buffer("support", support)
        self.register_buffer("centers", centers)

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

        self.dqn = ClassifierDQN(n_actions, cur_layer_img_w * cur_layer_img_h * 64, num_bins=num_bins, hidden_size=hidden_size)

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

        x = self.dqn(x)

        return x


    @property
    def device(self):
        return next(self.parameters()).device


    @torch.inference_mode()
    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        qvalues = self(states) 
        qvalues = torch.sum(F.softmax(qvalues, dim=-1) * self.centers, dim=-1)

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
