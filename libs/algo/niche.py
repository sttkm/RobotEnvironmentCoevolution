import os
import csv
import time
from itertools import chain
import pandas as pd
import numpy as np


class Niche:
    def __init__(self, key, parent_key, save_path):
        self.key = key
        self.parent_key = parent_key

        self.environment = None
        self.agents = {}

        self.recent_rewards = {}

        self.rewards_tmp = {}
        self.results_tmp = {}
        self.imigrants_tmp = {}

        self.best_rewards = {}
        self.history = pd.DataFrame()

        self.save_path = save_path

    def archive(self):
        # self.save_path = os.path.join(save_path, f'{self.key}')
        history_filename = os.path.join(self.save_path, 'history.csv')
        self.history.to_csv(history_filename)

        best_filename = os.path.join(self.save_path, 'best.csv')
        best = pd.DataFrame()
        for agent_key in sorted(self.best_rewards.keys()):
            reward = self.best_rewards[agent_key]
            best.loc[agent_key, 'reward'] = reward
        best.to_csv(best_filename)

        del self.recent_rewards
        del self.results_tmp
        del self.rewards_tmp
        del self.best_rewards
        del self.history
        del self.agents

    def reproduce(self, new_key, save_path, env_config):
        new_env_key, new_environment = env_config.reproduce(parent=self.environment)

        niche_path = os.path.join(save_path, str(new_key))
        new_niche = Niche(new_key, self.key, niche_path)
        new_niche.set_environment(new_environment)

        return new_niche

    def set_environment(self, environment):
        self.environment = environment
        environment.admitted(self.save_path)

    def set_agents(self, agents):
        self.agents = agents
        for agent_key,agent in agents.items():
            self.recent_rewards[agent_key] = []
            agent.set_pair_niche(self.key)

    def get_agent(self, agent_key):
        assert agent_key in self.agents
        return self.agents[agent_key]

    def clear_agents(self):
        for agent_key in list(self.agents.keys()):
            agent = self.agents.pop(agent_key)
            agent.drop_pair_niche(self.key)

    def end_iteration(self, iteration):
        self.imigrants_tmp = {}
        self.results_tmp = {}
        self.rewards_tmp = {}

    def get_agent_keys(self):
        return list(self.agents.keys())

    def get_agents(self):
        return self.agents

    def get_pair_keys(self):
        return [(self.key, agent_key) for agent_key in sorted(self.agents.keys())]

    def get_rewards(self, pair_keys=None):
        if pair_keys == None:
            pair_keys = self.get_pair_keys()
        return {pair_key: self.rewards_tmp[pair_key] for pair_key in pair_keys}

    def get_baselines(self):
        baselines = {agent_key: np.max(self.recent_rewards[agent_key][-10:]) for agent_key in self.agents.keys()}
        return baselines

    def get_imigrant_rewards(self):
        return {pair_key: result['reward'] for pair_key,(result,_) in self.imigrants_tmp.items()}

    def get_best_reward(self):
        return max(self.best_rewards.items(), key=lambda z: z[1])

    def get_elites_reward(self, num):
        return np.mean(sorted(list(self.best_rewards.values()), reverse=True)[:num])

    def check_parent(self, threshold):
        baselines = [max(self.recent_rewards[agent_key][-10:]) for agent_key in self.agents.keys()]
        return np.max(baselines) > threshold

    def get_parent_agent(self, threshold):
        return {agent_key: agent for agent_key,agent in self.agents.items() if self.rewards_tmp[(self.key, agent_key)] > threshold}

    def _update_best(self, niche_key, agent_key, agent, reward, learn_result=None):
        agent_best_reward = self.best_rewards.get(agent_key, float('-inf'))
        if reward > agent_best_reward:
            self.best_rewards[agent_key] = reward
            agent.save_core(self.key, niche_key, learn_result=learn_result)

    def save_cores(self, iteration):
        for agent_key,agent in self.agents.items():
            agent.save_core(self.key, self.key, iteration=iteration)


    def _get_agent_evaluate_kwds(self, agent_config):
        return {(self.key, agent_key): agent.get_evaluate_kwds(self.key, agent_config) for agent_key,agent in self.agents.items()}

    def _get_agent_learn_kwds(self, agent_config, bootstrap=False, transfer=False):
        return {(self.key, agent_key): agent.get_learn_kwds(self.key, agent_config, bootstrap=bootstrap, transfer=transfer) for agent_key,agent in self.agents.items()}


    def get_evaluate(self, env_config, agent_config, niches=None):

        if niches == None:
            agent_kwds = self._get_agent_evaluate_kwds(agent_config)

        else:
            agent_kwds_list = [niche._get_agent_evaluate_kwds(agent_config) for niche in niches.values()]
            agent_kwds = dict(chain.from_iterable(d.items() for d in agent_kwds_list))

        env_kwds = self.environment.get_evaluate_kwds(env_config)

        kwds_dict = {}
        for pair_key,agent_kwds_ in agent_kwds.items():
            if pair_key in self.rewards_tmp:
                continue

            kwds = dict(**env_kwds, **agent_kwds_)
            kwds_dict[(self.key, pair_key)] = kwds
        return kwds_dict


    def set_evaluate(self, iteration, results, niches=None):

        if niches == None:
            niches = {self.key: self}

        for pair_key, result in results.items():
            reward = result['reward']

            niche_key, agent_key = pair_key
            if niche_key == self.key:
                self.recent_rewards[agent_key].append(reward)
                self.history.loc[iteration, agent_key] = reward

            if niche_key == self.key:
                agent = self.get_agent(agent_key)
            else:
                agent = niches[niche_key].get_agent(agent_key)
            self._update_best(niche_key, agent_key, agent, reward)

            self.rewards_tmp[pair_key] = reward


    def get_learn(self, env_config, agent_config, niches=None, pair_keys=None, bootstrap=False, transfer=False):

        if niches == None:
            agent_kwds = self._get_agent_learn_kwds(agent_config, bootstrap=bootstrap)
            evaluate = False

        else:
            agent_kwds_list = [niche._get_agent_learn_kwds(agent_config, transfer=transfer) for niche in niches.values()]
            agent_kwds = dict(chain.from_iterable(d.items() for d in agent_kwds_list))
            agent_kwds = {pair_key: agent_kwds[pair_key] for pair_key in pair_keys}
            evaluate = True

        env_kwds = self.environment.get_learn_kwds(env_config)

        kwds_dict = {}
        for pair_key,agent_kwds_ in agent_kwds.items():
            if pair_key in self.imigrants_tmp or pair_key in self.results_tmp:
                continue
            kwds = dict(**env_kwds, **agent_kwds_, evaluate=evaluate)
            kwds_dict[(self.key, pair_key)] = kwds

        return kwds_dict

    def set_learn(self, iteration, results, niches=None):

        for pair_key, result in results.items():

            niche_key, agent_key = pair_key
            if niche_key == self.key:
                assert agent_key in self.agents
                self.results_tmp[agent_key] = result

            else:
                agent = niches[niche_key].get_agent(agent_key)
                self.imigrants_tmp[pair_key] = (result, agent)
                self._update_best(niche_key, agent_key, agent, result['reward'], learn_result=result)

    def set_learn_result(self):
        for agent_key, result in self.results_tmp.items():
            self.agents[agent_key].set_learn_result(self.key, result, None)

    def transferred(self, iteration, max_agent_num, agent_config):
        live_rewards = self.get_baselines()
        
        imigrants = {}
        imigrant_rewards = {}
        for pair_key, (result, agent) in self.imigrants_tmp.items():
            niche_key, agent_key = pair_key
            reward = result['reward']
            if agent_key in imigrants:
                reward_ = imigrants[agent_key][1]['reward']
                if reward > reward_:
                    imigrants[agent_key] = (niche_key, result, agent)
                    imigrant_rewards[agent_key] = reward
            else:
                imigrants[agent_key] = (niche_key, result, agent)
                imigrant_rewards[agent_key] = reward

        drop_keys = []
        accept_keys = []

        imigrant_rewards = sorted(imigrant_rewards.items(), key=lambda z: z[1], reverse=True)


        while len(imigrant_rewards) > 0 and len(live_rewards) - len(drop_keys) + len(accept_keys) < max_agent_num:
            imigrant_key, imigrant_reward = imigrant_rewards.pop(0)
            if imigrant_key in live_rewards:
                if imigrant_reward > live_rewards[imigrant_key]:
                    drop_keys.append(imigrant_key)
                    accept_keys.append(imigrant_key)
                    live_rewards.pop(imigrant_key)
                else:
                    continue
            else:
                accept_keys.append(imigrant_key)

        while len(imigrant_rewards) > 0 and len(live_rewards) > 0:
            live_min = min(live_rewards.items(), key=lambda z: z[1])
            if imigrant_rewards[0][1] <= live_min[1]:
                break

            imigrant_key, imigrant_reward = imigrant_rewards.pop(0)
            if imigrant_key in live_rewards:
                if imigrant_reward > live_rewards[imigrant_key]:
                    drop_keys.append(imigrant_key)
                    accept_keys.append(imigrant_key)
                    live_rewards.pop(imigrant_key)
                else:
                    continue
            else:
                live_key, _ = live_min
                drop_keys.append(live_key)
                accept_keys.append(imigrant_key)
                live_rewards.pop(live_key)
        

        for drop_key in drop_keys:
            drop_agent = self.agents.pop(drop_key)
            drop_agent.drop_pair_niche(self.key)
            self.recent_rewards.pop(drop_key)

        accept_pairs = []
        for accept_key in accept_keys:
            niche_key, result, accept_agent = imigrants[accept_key]

            accept_agent.set_pair_niche(self.key)
            accept_agent.set_learn_result(self.key, result, agent_config, transfer=True)

            reward = result['reward']
            self.agents[accept_key] = accept_agent
            self.rewards_tmp[(self.key, accept_key)] = reward
            self.recent_rewards[accept_key] = [reward]
            self.history.loc[iteration, accept_key] = reward

            accept_pairs.append((niche_key, accept_key))

        assert len(self.agents) <= max_agent_num, 'live: ' + ', '.join(map(str,self.agents.keys())) + '  drops: ' + ', '.join(map(str, drop_keys)) + '  accepts: ' + ', '.join(map(str,accept_keys))

        return drop_keys, accept_pairs

    def get_pass_pair_keys(self, niches, transfer_limit):
        # baseline = min(self.get_baselines().values())
        pair_keys = list(chain.from_iterable([[pair_key for pair_key in niche.get_pair_keys()] for niche in niches.values()]))
        pair_rewards = self.get_rewards(pair_keys=pair_keys)
        # passed = sorted(pair_rewards.items(), key=lambda z: -z[1])[:transfer_limit]
        # pass_pair_keys = [pair_key for pair_key,reward in passed]

        pass_agents = {}
        for pair_key,reward in pair_rewards.items():
            niche_key, agent_key = pair_key
            if agent_key in pass_agents:
                reward_ = pass_agents[agent_key][1]
                if reward > reward_:
                    pass_agents[agent_key] = (niche_key, reward)
            else:
                pass_agents[agent_key] = (niche_key, reward)

        pass_agents = sorted(pass_agents.items(), key=lambda z: z[1][1], reverse=True)[:transfer_limit]
        pass_pair_keys = [(niche_key, agent_key) for agent_key,(niche_key,_) in pass_agents]

        return pass_pair_keys