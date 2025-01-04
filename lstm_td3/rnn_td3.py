import itertools
import os
import time
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from torch.optim import Adam

from lstm_td3.buffer import ReplayBuffer
from lstm_td3.env_wrapper.env import make_dmc_manipulator, make_bullet_task
from lstm_td3.env_wrapper.pomdp_wrapper import POMDPWrapper
from lstm_td3.models import MLPActorCritic
from lstm_td3.test.test_envs import make_test_env_pi
from lstm_td3.utils.logx import TensorBoardLogger
from lstm_td3.utils.tools import SequenceCellType


class RNN_TD3:
    def __init__(self,
             resume_exp_dir=None,
             env_name='', seed=0,
             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
             start_steps=10000,
             update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
             noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
             batch_size=100,
             max_hist_len=100,
             rnn_cell_type: SequenceCellType = SequenceCellType.LSTM,
             partially_observable=False,
             pomdp_type = 'remove_velocity',
             flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
             use_double_critic = True,
             use_target_policy_smooth = True,
             critic_mem_pre_lstm_hid_sizes=(128,),
             critic_mem_lstm_hid_sizes=(128,),
             critic_mem_after_lstm_hid_size=(128,),
             critic_cur_feature_hid_sizes=(128,),
             critic_post_comb_hid_sizes=(128,),
             critic_hist_with_past_act=False,
             actor_mem_pre_lstm_hid_sizes=(128,),
             actor_mem_lstm_hid_sizes=(128,),
             actor_mem_after_lstm_hid_size=(128,),
             actor_cur_feature_hid_sizes=(128,),
             actor_post_comb_hid_sizes=(128,),
             actor_hist_with_past_act=False,
             compile=False,
             logger_kwargs=dict(), save_freq=1):
        """
        Twin Delayed Deep Deterministic Policy Gradient (TD3)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                these should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to TD3.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            act_noise (float): Stddev for Gaussian exploration noise added to
                policy at training time. (At test time, no noise is added.)

            target_noise (float): Stddev for smoothing noise added to target
                policy.

            noise_clip (float): Limit for absolute value of target policy
                smoothing noise.

            policy_delay (int): Policy will only be updated once every
                policy_delay times for each update of the Q-networks.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """
        # setting constructor args to self
        self.resume_exp_dir = resume_exp_dir
        self.env_name = env_name
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.batch_size = batch_size
        self.max_hist_len = max_hist_len
        self.rnn_cell_type = rnn_cell_type
        self.partially_observable = partially_observable
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob
        self.use_double_critic = use_double_critic
        self.use_target_policy_smooth = use_target_policy_smooth
        self.critic_mem_pre_lstm_hid_sizes = critic_mem_pre_lstm_hid_sizes
        self.critic_mem_lstm_hid_sizes = critic_mem_lstm_hid_sizes
        self.critic_mem_after_lstm_hid_size = critic_mem_after_lstm_hid_size
        self.critic_cur_feature_hid_sizes = critic_cur_feature_hid_sizes
        self.critic_post_comb_hid_sizes = critic_post_comb_hid_sizes
        self.critic_hist_with_past_act = critic_hist_with_past_act
        self.actor_mem_pre_lstm_hid_sizes = actor_mem_pre_lstm_hid_sizes
        self.actor_mem_lstm_hid_sizes = actor_mem_lstm_hid_sizes
        self.actor_mem_after_lstm_hid_size = actor_mem_after_lstm_hid_size
        self.actor_cur_feature_hid_sizes = actor_cur_feature_hid_sizes
        self.actor_post_comb_hid_sizes = actor_post_comb_hid_sizes
        self.actor_hist_with_past_act = actor_hist_with_past_act
        self.compile = compile

        # If not going to resume, create new logger.
        if resume_exp_dir is None:
            self.logger = TensorBoardLogger(**logger_kwargs)
            self.logger.save_config(locals())
        else:
            self.logger = TensorBoardLogger(**logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)

        if env_name.startswith("DMC"):
            env_name_no_domain = env_name.removeprefix("DMC_")
            task, variant = env_name_no_domain.split("_", 1)
            self.env = make_dmc_manipulator(task, variant, seed=seed)
            self.test_env = make_dmc_manipulator(task, variant, seed=seed)
        elif env_name.startswith("test"):
            self.env = make_test_env_pi()
            self.test_env = make_test_env_pi()
        else:
            # Wrapper environment if using POMDP
            if partially_observable:
                self.env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
                self.test_env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
            else:
                # env, test_env = gym.make(env_name), gym.make(env_name)
                self.env = make_bullet_task(env_name, dp_type='MDP')
                self.test_env = make_bullet_task(env_name, dp_type='MDP')
                self.env.seed(seed)
                self.test_env.seed(seed)

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, rnn_cell_type, self.act_limit,
                                 critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                                 critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                                 critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                                 critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                                 critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                                 critic_hist_with_past_act=critic_hist_with_past_act,
                                 actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                                 actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                                 actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                                 actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                                 actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                                 actor_hist_with_past_act=actor_hist_with_past_act)
        self.ac_targ = deepcopy(self.ac)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac.to(self.device)
        self.ac_targ.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, max_size=replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)
        # # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1, q1_extracted_memory = self.ac.q1(o, a, h_o, h_a, h_o_len)
        q2, q2_extracted_memory = self.ac.q2(o, a, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = self.ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            if self.use_target_policy_smooth:
                epsilon = torch.randn_like(pi_targ) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -self.act_limit, self.act_limit)
            else:
                a2 = pi_targ

            # Target Q-values
            q1_pi_targ, _ = self.ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ, _ = self.ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            if self.use_double_critic:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            else:
                q_pi_targ = q1_pi_targ
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if self.use_double_critic:
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = loss_q1

        # Useful info for logging
        # import pdb; pdb.set_trace()
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = self.ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = self.ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=-1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info

    @property
    def update(self):
        _update_val = getattr(self, "_update_val", None)
        if _update_val is not None:
            return _update_val
        if self.compile:
            update = torch.compile(self._update, mode="reduce-overhead")
        else:
            update = self._update
        self._update_val = update
        return self._update_val


    def _update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            self.logger.store(LossPi=loss_pi.item(), **loss_info_pi)

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, o_buff, a_buff, o_buff_len, noise_scale, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(self.act_dim)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            if self.max_hist_len > 0:
                o_buff = np.zeros([self.max_hist_len, self.obs_dim])
                a_buff = np.zeros([self.max_hist_len, self.act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, self.obs_dim])
                a_buff = np.zeros([1, self.act_dim])
                o_buff_len = 0

            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = self.get_action(o, o_buff, a_buff, o_buff_len, 0, self.device)
                o2, r, d, _ = self.test_env.step(a)

                ep_ret += r
                if hasattr(self.test_env, 'action_repeat'):
                    ep_len += self.test_env.action_repeat
                else:
                    ep_len += 1
                # Add short history
                if self.max_hist_len != 0:
                    if o_buff_len == self.max_hist_len:
                        o_buff[:self.max_hist_len - 1] = o_buff[1:]
                        a_buff[:self.max_hist_len - 1] = a_buff[1:]
                        o_buff[self.max_hist_len - 1] = list(o)
                        a_buff[self.max_hist_len - 1] = list(a)
                    else:
                        o_buff[o_buff_len + 1 - 1] = list(o)
                        a_buff[o_buff_len + 1 - 1] = list(a)
                        o_buff_len += 1
                o = o2

            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

        # Prepare for interaction with environment
    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        past_t = 0
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        if self.max_hist_len > 0:
            o_buff = np.zeros([self.max_hist_len, self.obs_dim])
            a_buff = np.zeros([self.max_hist_len, self.act_dim])
            o_buff[0, :] = o
            o_buff_len = 0
        else:
            o_buff = np.zeros([1, self.obs_dim])
            a_buff = np.zeros([1, self.act_dim])
            o_buff_len = 0

        if self.resume_exp_dir is not None:
            # Find the latest checkpoint
            resume_checkpoint_path = osp.join(self.resume_exp_dir, "pyt_save")
            checkpoint_files = os.listdir(resume_checkpoint_path)
            latest_context_version = np.max([int(f_name.split('-')[3]) for f_name in checkpoint_files if
                                             'context' in f_name and 'verified' in f_name])
            latest_model_version = np.max([int(f_name.split('-')[3]) for f_name in checkpoint_files if
                                           'model' in f_name and 'verified' in f_name])
            if latest_context_version != latest_model_version:
                latest_version = np.min([latest_context_version, latest_model_version])
            else:
                latest_version = latest_context_version
            latest_context_checkpoint_file_name = 'checkpoint-context-Step-{}-verified.pt'.format(latest_version)
            latest_model_checkpoint_file_name = 'checkpoint-model-Step-{}-verified.pt'.format(latest_version)
            latest_context_checkpoint_file_path = osp.join(resume_checkpoint_path, latest_context_checkpoint_file_name)
            latest_model_checkpoint_file_path = osp.join(resume_checkpoint_path, latest_model_checkpoint_file_name)

            # Load the latest checkpoint
            context_checkpoint = torch.load(latest_context_checkpoint_file_path)
            model_checkpoint = torch.load(latest_model_checkpoint_file_path)

            # Restore experiment context
            self.logger.epoch_dict = context_checkpoint['logger_epoch_dict']
            self.replay_buffer = context_checkpoint['replay_buffer']
            self.start_time = context_checkpoint['start_time']
            self.past_t = context_checkpoint['t'] + 1  # Crucial add 1 step to t to avoid repeating.

            # Restore model
            self.ac.load_state_dict(model_checkpoint['ac_state_dict'])
            self.ac_targ.load_state_dict(model_checkpoint['target_ac_state_dict'])
            self.pi_optimizer.load_state_dict(model_checkpoint['pi_optimizer_state_dict'])
            self.q_optimizer.load_state_dict(model_checkpoint['q_optimizer_state_dict'])

        print("past_t={}".format(past_t))
        # Main loop: collect experience in env and update/log each epoch
        for t in range(past_t, total_steps):  # Start from the step after resuming.
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if t > self.start_steps:
                a = self.get_action(o, o_buff, a_buff, o_buff_len, self.act_noise, self.device)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)

            ep_ret += r
            if hasattr(self.env, 'action_repeat'):
                ep_len += self.env.action_repeat
            else:
                ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Add short history
            if self.max_hist_len != 0:
                if o_buff_len == self.max_hist_len:
                    o_buff[:self.max_hist_len - 1] = o_buff[1:]
                    a_buff[:self.max_hist_len - 1] = a_buff[1:]
                    o_buff[self.max_hist_len - 1] = list(o)
                    a_buff[self.max_hist_len - 1] = list(a)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(o)
                    a_buff[o_buff_len + 1 - 1] = list(a)
                    o_buff_len += 1

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

                if self.max_hist_len > 0:
                    o_buff = np.zeros([self.max_hist_len, self.obs_dim])
                    a_buff = np.zeros([self.max_hist_len, self.act_dim])
                    o_buff[0, :] = o
                    o_buff_len = 0
                else:
                    o_buff = np.zeros([1, self.obs_dim])
                    a_buff = np.zeros([1, self.act_dim])
                    o_buff_len = 0

                # Store checkpoint at the end of trajectory, so there is no need to store env as resume env in PyBullet is problematic.
                # Save the context of the learning and learned models
                fpath = 'pyt_save'
                fpath = osp.join(self.logger.output_dir, fpath)
                os.makedirs(fpath, exist_ok=True)
                old_checkpoints = os.listdir(fpath)  # Cache old checkpoints to delete later
                # Separately save context and model to reduce disk space usage.
                context_fname = 'checkpoint-context-' + (
                    'Step-%d' % t if t is not None else '') + '.pt'
                model_fname = 'checkpoint-model-' + ('Step-%d' % t if t is not None else '') + '.pt'

                context_elements = {'env': self.env, 'replay_buffer': self.replay_buffer,
                                    'logger_epoch_dict': self.logger.epoch_dict,
                                    'start_time': start_time, 't': t}
                model_elements = {'ac_state_dict': self.ac.state_dict(),
                                  'target_ac_state_dict': self.ac_targ.state_dict(),
                                  'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                                  'q_optimizer_state_dict': self.q_optimizer.state_dict()}
                context_fname = osp.join(fpath, context_fname)
                torch.save(context_elements, context_fname)
                model_fname = osp.join(fpath, model_fname)
                torch.save(model_elements, model_fname)
                # Rename the file to verify the completion of the saving.
                verified_context_fname = osp.join(fpath, 'checkpoint-context-' + (
                    'Step-%d' % t if t is not None else '') + '-verified.pt')
                verified_model_fname = osp.join(fpath, 'checkpoint-model-' + (
                    'Step-%d' % t if t is not None else '') + '-verified.pt')
                os.rename(context_fname, verified_context_fname)
                os.rename(model_fname, verified_model_fname)
                # Remove old checkpoint
                for old_f in old_checkpoints:
                    os.remove(osp.join(fpath, old_f))

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch_with_history(self.batch_size, self.max_hist_len)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    torch.compiler.cudagraph_mark_step_begin()
                    self.update(data=batch, timer=j)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                self.logger.log_tabular('scalars/Epoch', epoch, timestep=epoch)
                self.logger.log_tabular('scalars/EpRet', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/TestEpRet', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/EpLen', average_only=True, timestep=epoch)
                self.logger.log_tabular('scalars/TestEpLen', average_only=True, timestep=epoch)
                self.logger.log_tabular('scalars/TotalEnvInteracts', t, timestep=epoch)
                self.logger.log_tabular('scalars/Q1Vals', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/Q2Vals', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/Q1ExtractedMemory', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/Q2ExtractedMemory', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/ActExtractedMemory', with_min_and_max=True, timestep=epoch)
                self.logger.log_tabular('scalars/LossPi', average_only=True, timestep=epoch)
                self.logger.log_tabular('scalars/LossQ', average_only=True, timestep=epoch)

                self.logger.log_tabular('scalars/Time', time.time() - start_time, timestep=epoch)
                self.logger.dump_tabular()
