# testing saved model
import os
import re

import torch

from lstm_td3.rnn_td3 import RNN_TD3
from lstm_td3.utils.logx import setup_logger_kwargs
from lstm_td3.utils.tools import str2bool

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_dir', type=str, default=None, help="The directory of the resuming experiment.")
    parser.add_argument('--env_name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--max_hist_len', type=int, default=5)
    parser.add_argument('--partially_observable', type=str2bool, nargs='?', const=True, default=False, help="Using POMDP")
    parser.add_argument('--pomdp_type',
                        choices=['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing',
                                 'remove_velocity_and_flickering', 'remove_velocity_and_random_noise',
                                 'remove_velocity_and_random_sensor_missing', 'flickering_and_random_noise',
                                 'random_noise_and_random_sensor_missing', 'random_sensor_missing_and_random_noise'],
                        default='remove_velocity')
    parser.add_argument('--rnn_cell_type', choices=['lstm', 'cfc', 'ltc'], default='lstm')
    parser.add_argument('--flicker_prob', type=float, default=0.2)
    parser.add_argument('--random_noise_sigma', type=float, default=0.1)
    parser.add_argument('--random_sensor_missing_prob', type=float, default=0.1)
    parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True,
                        help="Using double critic")
    parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True,
                        help="Using target policy smoothing")
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--critic_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--critic_cur_feature_hid_sizes', type=int, nargs="?", default=[64, 64])
    parser.add_argument('--critic_post_comb_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--critic_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--actor_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--actor_cur_feature_hid_sizes', type=int, nargs="?", default=[64, 64])
    parser.add_argument('--actor_post_comb_hid_sizes', type=int, nargs="+", default=[64])
    parser.add_argument('--actor_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--exp_name', type=str, default='lstm_td3')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm_gate')
    parser.add_argument("--compile", type=str2bool, default=False)
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=args.data_dir, datestamp=False)
    model_path = os.path.join('logdir', logger_kwargs['output_dir'], 'pyt_save')

    algo = RNN_TD3(resume_exp_dir=args.resume_exp_dir,
                   env_name=args.env_name,
                   gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                   max_hist_len=args.max_hist_len,
                   partially_observable=args.partially_observable,
                   pomdp_type=args.pomdp_type,
                   flicker_prob=args.flicker_prob,
                   random_noise_sigma=args.random_noise_sigma,
                   random_sensor_missing_prob=args.random_sensor_missing_prob,
                   use_double_critic=args.use_double_critic,
                   use_target_policy_smooth=args.use_target_policy_smooth,
                   rnn_cell_type=args.rnn_cell_type,
                   critic_mem_pre_lstm_hid_sizes=tuple(args.critic_mem_pre_lstm_hid_sizes),
                   critic_mem_lstm_hid_sizes=tuple(args.critic_mem_lstm_hid_sizes),
                   critic_mem_after_lstm_hid_size=tuple(args.critic_mem_after_lstm_hid_size),
                   critic_cur_feature_hid_sizes=tuple(args.critic_cur_feature_hid_sizes),
                   critic_post_comb_hid_sizes=tuple(args.critic_post_comb_hid_sizes),
                   critic_hist_with_past_act=args.critic_hist_with_past_act,
                   actor_mem_pre_lstm_hid_sizes=tuple(args.actor_mem_pre_lstm_hid_sizes),
                   actor_mem_lstm_hid_sizes=tuple(args.actor_mem_lstm_hid_sizes),
                   actor_mem_after_lstm_hid_size=tuple(args.actor_mem_after_lstm_hid_size),
                   actor_cur_feature_hid_sizes=tuple(args.actor_cur_feature_hid_sizes),
                   actor_post_comb_hid_sizes=tuple(args.actor_post_comb_hid_sizes),
                   actor_hist_with_past_act=args.actor_hist_with_past_act,
                   compile=args.compile,
                   logger_kwargs=logger_kwargs)

    algo.load(model_path)
    algo.test_agent(0, inference=True)


