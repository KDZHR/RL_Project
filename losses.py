# https://arxiv.org/abs/1806.04613
import torch
import torch.special
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float, device):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32, device=device
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print(logits.shape)
        print(target.shape)
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
        (self.support - target.unsqueeze(-1))
        / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_agent,
                    gamma=0.99,
                    check_shapes=False,
                    device="cpu"):
    """ Compute td loss using torch operations only. Use the formulae above. """
    batch_size = len(states)
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"

    # compute q-values for all actions in next states
    with torch.no_grad():
        predicted_next_qvalues = agent(next_states)
        target_predicted_next_qvalues = target_agent(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[np.arange(batch_size), actions]

    # compute V*(next_states) using predicted next q-values
    next_state_values = predicted_next_qvalues.argmax(dim=1)  # we actually don't compute V*(s') here because we use qvalues from target_network

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * target_predicted_next_qvalues[np.arange(batch_size), next_state_values] * is_not_done

    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"

    # mean squared error loss to minimize
    loss = ((predicted_qvalues_for_actions - target_qvalues_for_actions) ** 2).mean()

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


def compute_td_ce_loss(states, actions, rewards, next_states, is_done,
                    agent, target_agent,
                    min_value: float, max_value: float, num_bins: int, sigma: float,
                    gamma=0.99,
                    check_shapes=False,
                    device="cpu"):
    batch_size = len(states)
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_q_dist = agent(states) # [batch_size, n_actions, n_bins]
    # compute q-values for all actions in next states
    with torch.no_grad():
        predicted_next_qvalues = torch.tensor(agent.get_qvalues(next_states), device=device)  # [batch_size, n_actions, n_bins]
        target_predicted_next_qvalues = torch.tensor(target_agent.get_qvalues(next_states), device=device)  # [batch_size, n_actions, n_bins]

    # select q-values for chosen actions
    predicted_q_dist_for_actions = predicted_q_dist[np.arange(batch_size), actions]

    next_state_values = predicted_next_qvalues.argmax(dim=1)

    target_qvalues_for_actions = rewards + gamma * target_predicted_next_qvalues[np.arange(batch_size), next_state_values] * is_not_done

    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"

    # mean squared error loss to minimize
    hl_loss = HLGaussLoss(min_value=min_value, max_value=max_value, num_bins=num_bins, sigma=sigma, device=device)
    loss = hl_loss(predicted_q_dist_for_actions, target_qvalues_for_actions)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, "there's something wrong with target q-values, they must be a vector"

    return loss