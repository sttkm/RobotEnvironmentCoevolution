import os, sys
import copy
from itertools import count
from collections import OrderedDict
import numpy as np
import torch
import pandas as pd
import random
import time

import matplotlib.pyplot as plt


from gym_utils import make_vec_envs

from ppo import PPO, Policy

from evogym import get_full_connectivity
import neat_cppn

class EvogymStructureDecoder:
    def __init__(self, size):
        self.size = size

        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
        x = x.flatten()
        y = y.flatten()

        center = (np.array(size) - 1) / 2
        d = np.sqrt(np.square(x - center[0]) + np.square(y - center[1]))

        self.inputs = np.vstack([x, y, d]).T

    def decode(self, genome, config):
        cppn = neat_cppn.FeedForwardNetwork.create(genome, config)

        states = []
        for inp in self.inputs:
            state = cppn.activate(inp)
            states.append(state)

        output = np.vstack(states)
        output[:, 0] += 0.3
        material = np.argmax(output, axis=1)

        body = np.reshape(material, self.size)
        connections = get_full_connectivity(body)
        return (body, connections)

from evogym import is_connected, has_actuator, hashable

class EvogymStructureConstraint:
    def __init__(self, decode_function):
        self.decode_function = decode_function
        self.hashes = {}
        self.hashes_tmp = {}

    def has_actuator(self, body):
        actuator_count = np.sum(body>=3)
        return actuator_count >= 5

    def check_density(self, body):
        voxel_count = np.sum(body>0)
        return voxel_count >= 5

    def eval_constraint(self, genome, config, *args):
        robot = self.decode_function(genome, config)
        body,_ = robot
        validity = is_connected(body) and self.has_actuator(body) and self.check_density(body)
        if validity:
            robot_hash = hashable(body)
            if robot_hash in self.hashes or robot_hash in self.hashes_tmp:
                validity = False
            else:
                self.hashes[robot_hash] = True

        return validity

    def add_hash(self, robot):
        # body,_ = robot
        # robot_hash = hashable(body)
        # self.hashes[robot_hash] = True
        pass

    def clear_tmp(self):
        self.hashes_tmp = {}



def evaluate(env_id, robot, terrain, policy_params=None, obs_rms=None):
    envs = make_vec_envs(env_id, robot, terrain, 1, subproc=False, vecnorm=True)
    envs.training = False

    policy = Policy(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
    )
    if policy_params is not None:
        policy.load_state_dict(policy_params)
    if obs_rms is not None:
        envs.obs_rms = obs_rms

    obs = envs.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = policy.predict(obs, deterministic=True)
        obs, _, done, infos = envs.step(action)

        if 'episode' in infos[0]:
            reward = infos[0]['episode']['r']

    envs.close()

    del envs

    result = {'reward': reward}

    return result

def learn(env_id, robot, terrain, ppo_kwargs, num_processes, steps, policy_params=None, opt_params=None, obs_rms=None, lr=2.5e-4, evaluate=True):
    
    train_envs = make_vec_envs(env_id, robot, terrain, num_processes, subproc=False, vecnorm=True)

    policy = Policy(
        observation_space=train_envs.observation_space,
        action_space=train_envs.action_space,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    ppo = PPO(train_envs, **ppo_kwargs)

    if policy_params is not None:
        policy.load_state_dict(policy_params)
    if opt_params is not None:
        optimizer.load_state_dict(opt_params)
    if obs_rms is not None:
        train_envs.obs_rms = obs_rms

    for _ in range(steps):
        ppo.step(policy, optimizer)

    result = {
        'policy_params': policy.state_dict(),
        'optimizer_params': optimizer.state_dict(),
        'obs_rms': train_envs.obs_rms
        # 'obs_rms': None
    }

    if evaluate:
        eval_env = make_vec_envs(env_id, robot, terrain, 1, subproc=False, vecnorm=True)
        eval_env.training = False

        if obs_rms is not None:
            eval_env.obs_rms = train_envs.obs_rms

        obs = eval_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = policy.predict(obs, deterministic=True)
            obs, _, done, infos = eval_env.step(action)

            if 'episode' in infos[0]:
                reward = infos[0]['episode']['r']
                result['reward'] = reward

        eval_env.close()
        del eval_env

    train_envs.close()
    del train_envs

    return result

    


class Agent:
    def __init__(self, key, cppn_genome, parent_key, save_path):
        self.key = key
        self.cppn_genome = cppn_genome

        self.parent_key = parent_key

        self.robot = None

        self.cores = {}
        self.pair_niche_keys = []

        self.save_path = save_path
        self.core_path = os.path.join(self.save_path, 'core')
        # self.history = pd.DataFrame()

    def make_robot(self, decoder, genome_config):
        self.robot = decoder.decode(self.cppn_genome, genome_config)

    def admitted(self, config):
        config.add_robot_hashe(self.robot)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_robot(config)
        os.makedirs(self.core_path, exist_ok=True)

    def save_robot(self, config):
        robot_file = os.path.join(self.save_path, 'robot.npz')
        body, connections = self.robot
        np.savez(robot_file, body, connections)
        self.draw_robot_image(body, config)

        filename = os.path.join(self.save_path, 'robot.txt')
        with open(filename, 'w') as f:
            for i in range(body.shape[0]):
                row = body[i]
                row_text = ' '.join([str(int(r)) for r in row])
                f.write(row_text + '\n')

    def draw_robot_image(self, body, config):
        image_file = os.path.join(self.save_path, 'robot.jpg')
        size = body.shape

        voxel_colors = [
            (0.96, 0.96, 0.96),
            (0.15, 0.15, 0.15),
            (0.75, 0.75, 0.75),
            (0.93, 0.57, 0.30),
            (0.47, 0.67, 0.82)
        ]
        fig, ax = plt.subplots(figsize=(size[1], size[0]))
        for y_ in range(size[0]):
            y = size[0] - y_ - 1
            for x in range(size[1]):
                voxel = body[y_, x]
                ax.fill_between([x,x+1], [y+1, y+1], [y, y], fc=voxel_colors[voxel])

        ax.set_xlim([0, size[1]])
        ax.set_ylim([0, size[0]])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.savefig(image_file, bbox_inches='tight')
        plt.close()


    def set_pair_niche(self, niche_key):
        assert niche_key not in self.pair_niche_keys
        self.pair_niche_keys.append(niche_key)

    def drop_pair_niche(self, niche_key):
        assert niche_key in self.cores and niche_key in self.pair_niche_keys
        self.cores.pop(niche_key)
        self.pair_niche_keys.remove(niche_key)


    def get_evaluate_kwds(self, niche_key, config):
        kwds = self.cores[niche_key]
        kwds = {
            'robot': self.robot, 
            'policy_params': self.cores[niche_key]['policy_params'],
            'obs_rms': self.cores[niche_key]['obs_rms']
        }
        return kwds

    def get_learn_kwds(self, niche_key, config, bootstrap=False, transfer=False):
        if niche_key in self.cores:
            policy_params = self.cores[niche_key]['policy_params']
            opt_params = self.cores[niche_key]['opt_params']
            obs_rms = self.cores[niche_key]['obs_rms']
        else:
            policy_params = None
            opt_params = None
            obs_rms = None

        if bootstrap:
            steps = config.steps_bootstrap
            ppo_kwargs = dict(**config.ppo_kwargs)
            ppo_kwargs['learning_rate'] = 5e-4
            # ppo_kwargs['n_steps'] = 128
            # ppo_kwargs['batch_size'] = ppo_kwargs['batch_size'] // 2
            ppo_kwargs['n_epochs'] = 8
            ppo_kwargs['clip_range'] = 0.3
            ppo_kwargs['clip_range_vf'] = 0.3
        elif transfer:
            steps = config.steps_transfer
            # ppo_kwargs = config.ppo_kwargs
            ppo_kwargs = dict(**config.ppo_kwargs)
            ppo_kwargs['learning_rate'] = 3e-4
            # ppo_kwargs['n_steps'] = 128
            # ppo_kwargs['batch_size'] = ppo_kwargs['batch_size'] // 2
            ppo_kwargs['n_epochs'] = 6
            ppo_kwargs['clip_range'] = 0.2
            ppo_kwargs['clip_range_vf'] = 0.2
        else:
            steps = config.steps_per_iteration
            ppo_kwargs = config.ppo_kwargs
            
        kwds = {
            'robot': self.robot,
            'policy_params': policy_params,
            'opt_params': opt_params,
            'obs_rms': obs_rms,
            'num_processes': config.num_processes,
            'steps': steps,
            'ppo_kwargs': ppo_kwargs,
        }
        return kwds
        
    def set_learn_result(self, niche_key, result, config, transfer=False):
        policy_params = result['policy_params']
        opt_params = result['optimizer_params']
        obs_rms = result['obs_rms']

        if transfer:
            action_size = policy_params['log_std'].shape
            policy_params['log_std'] = torch.full(action_size, config.init_log_std)
            for key,state in opt_params['state'].items():
                opt_params['state'][key]['step'] = min(state['step'], 2**16)

        self.cores[niche_key] = {
            'policy_params': policy_params,
            'opt_params': opt_params,
            'obs_rms': obs_rms
        }

    def save_core(self, target_niche_key, niche_key, learn_result=None, iteration=None):
        assert niche_key in self.cores

        if learn_result is None:
            policy_params = self.cores[niche_key]['policy_params']
            obs_rms = self.cores[niche_key]['obs_rms']
        else:
            policy_params = learn_result['policy_params']
            obs_rms = learn_result['obs_rms']

        if iteration is None:
            filename = os.path.join(self.core_path, f'{target_niche_key}.pt')
        else:
            filename = os.path.join(self.core_path, f'{target_niche_key}_{iteration}.pt')
        
        torch.save([policy_params, obs_rms], filename)

    def get_core(self, niche_key):
        return self.cores[niche_key]



class AgentConfig:
    def __init__(self,
                 robot_size,
                 neat_config,
                 save_path,
                 steps_per_iteration=4,
                 steps_bootstrap=120,
                 steps_transfer=20,
                 clip_range=0.1,
                 epochs=4,
                 num_mini_batch=4,
                 steps=128,
                 num_processes=4,
                 gamma=0.99,
                 learning_rate=2.5e-4,
                 gae_lambda=0.95,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 lr_decay=False,
                 init_log_std=0.0,
                 max_steps=2000):

        self.neat_config = neat_config
        self.save_path = save_path

        self.decoder = EvogymStructureDecoder(robot_size)
        self.constraint = EvogymStructureConstraint(self.decoder.decode)

        self.cppn_indexer = count(0)
        self.agent_indexer = count(0)

        self.steps_per_iteration = steps_per_iteration
        self.steps_bootstrap = steps_bootstrap
        self.steps_transfer = steps_transfer
        self.num_processes = num_processes
        self.learning_rate = learning_rate
        self.init_log_std = init_log_std

        self.ppo_kwargs = {
            'learning_rate': learning_rate,
            'n_steps': steps,
            'batch_size': steps * num_processes // num_mini_batch // 2,
            'n_epochs': epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'clip_range_vf': clip_range,
            'normalize_advantage': True,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'device': 'cpu',
            'lr_decay': lr_decay,
            'max_iter': max_steps,
            'iter': 0
        }

        os.makedirs(self.save_path, exist_ok=True)

    def make_init(self):
        key, agent = self.reproduce()
        return key, agent

    def reproduce_cppn_genome(self, parent=None):
        key = next(self.cppn_indexer)
        invalid = True
        while invalid:
            if parent is None:
                genome = self.neat_config.genome_type(key)
                genome.configure_new(self.neat_config.genome_config)
            else:
                genome = copy.deepcopy(parent)
                genome.mutate(self.neat_config.genome_config)
                genome.key = key
            invalid = not self.constraint.eval_constraint(genome, self.neat_config.genome_config)
        return genome

    def reproduce(self, parent=None):
        key = next(self.agent_indexer)
        if parent is None or random.random() < 0.1:
            parent_key = -1
            cppn_genome = self.reproduce_cppn_genome()
        else:
            parent_key = parent.key
            cppn_genome = self.reproduce_cppn_genome(parent=parent.cppn_genome)

        agent_path = os.path.join(self.save_path, str(key))
        agent = Agent(key, cppn_genome, parent_key, agent_path)
        agent.make_robot(self.decoder, self.neat_config.genome_config)
        agent.admitted(self)
        return key, agent

    def add_robot_hashe(self, robot):
        self.constraint.add_hash(robot)


