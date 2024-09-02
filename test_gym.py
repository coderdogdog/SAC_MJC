# -*- coding: utf-8 -*-
# @env     : 
# @Python  : 
# @Time    : 2024/9/2 13:46
# @Author  : Henry_Nan
# @File    : test_gym.py
# @Project : SAC_MJC


import gym
import numpy as np
import torch
from SAC import SAC_Agent


def Action_adapter(a, max_action=0.4):
    return a * max_action


def Action_adapter_reverse(a, max_action=0.4):
    return a / max_action


def evaluate_policy(name_env, model, max_action, test_num=3, xr=False):
    scores = 0
    turns = test_num
    if xr:
        env1 = gym.make(name_env, render_mode="human")
    else:
        env1 = gym.make(name_env, render_mode="rgb_array")
    env1 = env1.unwrapped  # 不可或缺

    for j in range(turns):
        s, _ = env1.reset()
        done, ep_r = False, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)     # [0,1] to [-max,max]
            s_prime, r, termi, trun, _ = env1.step(act)
            done = termi | trun
            ep_r += r
            s = s_prime
            if xr:
                env1.render()
        # print(ep_r)
        scores += ep_r
    return scores / turns


kwargs = {
    "state_dim": 11,  # state_dim
    "action_dim": 3,  # action_dim
    "gamma": 0.99,
    "hid_dim": 128,
    "a_lr": 3e-4,
    "c_lr": 3e-4,
    "batch_size": 256,
    "alpha": 0.12,
    "adaptive_alpha": True,
}

model = SAC_Agent(**kwargs)
model.load(170000)
env_name = 'Hopper-v4'

average_reward = evaluate_policy(env_name, model, 1.0, 5, True)
print('Average Reward:', average_reward)
