from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt

from dqn import DQNAgent
from replay_buffer import ReplayBuffer
import utils
import random
import numpy as np
import torch
import torch.nn as nn

from metrics import evaluate
from losses import compute_td_loss

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import cv2
import atari_wrappers
import gymnasium as gym

from framebuffer import FrameBuffer


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        super().__init__(env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)


    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        return np.dot(rgb[...,:3], channel_weights)[None, :, :]

    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        img = img[31:-17, 7:-8] # crop
        img = cv2.resize(img, (64, 64)) # resize
        img = self._to_gray_scale(img) # grayscale
        img = img.astype(np.float32) / 256 # float
        return img


def PrimaryAtariWrap(env, clip_rewards=True):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = agent.sample_actions(np.array([s]))[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, terminated)
        sum_rewards += r
        s = next_s
        if terminated or truncated:
            s, _ = env.reset()

    return sum_rewards, s


ENV_NAME = "ALE/Breakout-v5"
def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME, render_mode="rgb_array")  # create raw env
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


def train(config):
    timesteps_per_epoch = config["timesteps_per_epoch"]
    batch_size = config["batch_size"]
    total_steps = config["total_steps"]
    decay_steps = config["decay_steps"]
    lr = config["lr"]
    init_epsilon = config["init_epsilon"]
    final_epsilon = config["final_epsilon"]
    loss_freq = config["loss_freq"]
    refresh_target_network_freq = config["refresh_target_network_freq"]
    eval_freq = config["eval_freq"]
    max_grad_norm = config["max_grad_norm"]
    device = config["device"]
    n_lives = config["n_lives"]
    seed = config["seed"]
    stop_after_n_steps = config["stop_after_n_steps"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state, _ = env.reset()

    agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_shape, n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())

    print(f"Total parameters: {sum(p.numel() for p in agent.parameters())}")
    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []

    exp_replay = ReplayBuffer(10**5)
    for i in range(100):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available.
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """)
            break
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(len(exp_replay))

    state, _ = env.reset()
    for step in trange(total_steps + 1):
        if step > stop_after_n_steps:
            break
        if not utils.is_enough_ram():
            print('less that 100 Mb RAM available, freezing')
            print('make sure everything is ok and make KeyboardInterrupt to continue')
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                pass

        agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)

        loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,
                            agent, target_network,
                            gamma=0.99,
                            device=device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            grad_norm_history.append(grad_norm.cpu().item())

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            mean_rw_history.append(evaluate(
                make_env(clip_rewards=True, seed=step), agent, n_games=3 * 3, greedy=True)
            )
            initial_state_q_values = agent.get_qvalues(
                np.array([make_env(seed=step).reset()[0]])
            )
            initial_state_v_history.append(np.max(initial_state_q_values))

            clear_output(True)
            print("buffer size = %i, epsilon = %.5f" %
                (len(exp_replay), agent.epsilon))

            plt.figure(figsize=[16, 9])

            plt.subplot(2, 2, 1)
            plt.title("Mean reward per life")
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(td_loss_history[-1])
            plt.subplot(2, 2, 2)
            plt.title("TD loss history (smoothened)")
            plt.plot(utils.smoothen(td_loss_history))
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.title("Initial state V")
            plt.plot(initial_state_v_history)
            plt.grid()

            plt.subplot(2, 2, 4)
            plt.title("Grad norm history (smoothened)")
            plt.plot(utils.smoothen(grad_norm_history))
            plt.grid()
            plt.show()

    return {
        "mean_rw_history": mean_rw_history,
        "td_loss_history": td_loss_history,
        "grad_norm_history": grad_norm_history,
        "initial_state_v_history": initial_state_v_history,
        "agent": agent,
        "target_network": target_network,
        "exp_replay": exp_replay,
        "env": env,
        "state_shape": state_shape,
        "n_actions": n_actions,
        "state": state,
        "opt": opt,
        "device": device,
    }