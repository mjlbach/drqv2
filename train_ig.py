import gym
import os
import igibson
import dreamerv2.api as dv2
from igibson.envs.behavior_reward_shaping_env import BehaviorRewardShapingEnv
# from igibson.envs.behavior_env import BehaviorEnv

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# taken from dmc_vision
config = dv2.defaults.update({
    'jit': True,
    'logdir': '~/logdir/igibson-plan2xplore',
    'log_every': 1e3,
    'log_keys_video': ['rgb'],
    'train_every': 10,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'task': 'igibson',
    'render_size': [128, 128],
    'encoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb|depth', 'cnn_kernels': (4, 4, 4, 4)  },
    'decoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb|depth', 'cnn_kernels': (4, 6, 6, 6, 10) },
    'action_repeat': 2,
    'eval_every': 1e4,
    'expl_behavior': 'Plan2Explore',
    'prefill': 100, # 1000 dmc_vision
    'pretrain': 100,
    'clip_rewards': 'identity',
    'pred_discount': False,
    'grad_heads': ['decoder', 'reward'],
    'rssm': {'hidden': 200, 'deter': 200},
    'model_opt': {'lr' : 3e-4},
    'actor_opt': {'lr' : 8e-5},
    'critic_opt': {'lr' : 8e-5},
    'actor_ent': 1e-4,
    'kl': {'free' : 1.0},
    'replay': {
        'capacity': 100,  #2e6 default
        'prioritize_ends': True
    }
}).parse_flags()

env_config = "behavior_full_observability.yaml"
env = BehaviorRewardShapingEnv(env_config)
# env = BehaviorEnv(env_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)
