"""
python main.py

tensorboard --logdir=runs

python main.py --EnvIdex 0 --write False --render True --Loadmodel True --ModelIdex 100

"""

import os
import shutil
from datetime import datetime

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ReplayBuffer import RandomBuffer
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


def main():

    is_train = True
    env_name = 'Hopper-v4'     # 'Humanoid-v4'

    if is_train:
        write = True
        Loadmodel = False
        ModelIdex = None
    else:
        write = False
        Loadmodel = True
        ModelIdex = 10000

    env = gym.make(env_name, render_mode="rgb_array")
    env = env.unwrapped

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = env.action_space.high[0]
    max_per_epoch = env.spec.max_episode_steps  # 1000
    print('Env: Humanoid-v4', '  state_dim:', state_dim, '  action_dim:', action_dim,
          '  max_a:', max_action, '  min_a:', -1 * max_action, '   max_episode_steps', max_per_epoch)

    # Random seed config:
    random_seed = 0
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Interaction config:
    start_steps = 5 * max_per_epoch  # in steps
    update_after = 2 * max_per_epoch  # in steps
    update_every = 50
    total_steps = int(5e6)
    eval_interval = int(1e3)
    save_interval = int(1e4)

    # SummaryWriter config:
    if write:
        timenow = str(datetime.now())[0:-10]
        datenow = timenow[0:10]
        hourtime = timenow[-5:-3]
        mintime = timenow[-2:]
        timenow = '_' + datenow + '_' + hourtime + '-' + mintime
        writepath = 'runs/SAC_{}'.format(env_name) + timenow

        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,  # state_dim
        "action_dim": action_dim,  # action_dim
        "gamma": 0.99,
        "hid_dim": 128,
        "a_lr": 3e-4,
        "c_lr": 3e-4,
        "batch_size": 256,
        "alpha": 0.12,
        "adaptive_alpha": True,
    }

    model = SAC_Agent(**kwargs)
    if not os.path.exists('model'):
        os.mkdir('model')
    if Loadmodel:
        model.load(ModelIdex)

    replay_buffer = RandomBuffer(state_dim, action_dim, True)

    if not is_train:
        env.close()
        average_reward = evaluate_policy(env_name, model, max_action, 5, True)
        print('Average Reward:', average_reward)
    else:
        s, _ = env.reset()
        done, current_steps = False, 0

        for t in range(total_steps):
            current_steps += 1
            '''Interact & trian'''

            if t < start_steps:
                # Random explore for start_steps
                act = env.action_space.sample()  # act∈[-max,max]
                a = Action_adapter_reverse(act, max_action)  # a∈[-1,1]
            else:
                a = model.select_action(s, deterministic=False)  # a∈[-1,1]
                act = Action_adapter(a, max_action)  # act∈[-max,max]
            # act 给环境， a 给环境

            s_prime, r, termi, trun, _ = env.step(act)
            done = termi | trun
            dead = done

            replay_buffer.add(s, a, r, s_prime, dead)
            s = s_prime

            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    model.train(replay_buffer)

            '''save model'''
            if (t + 1) % save_interval == 0:
                model.save(t + 1)

            '''record & log'''
            if (t + 1) % eval_interval == 0:
                score = evaluate_policy(env_name, model, max_action)
                if write:
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                print('EnvName:{}'.format(env_name), ' seed:', random_seed, ' totalsteps:', t + 1, ' score:', score)
            if done:
                s, _ = env.reset()
                done, current_steps = False, 0

    env.close()


if __name__ == '__main__':
    main()
