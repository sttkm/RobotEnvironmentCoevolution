
import os
import csv
import time

import random
import pickle
from tqdm import tqdm
from itertools import product, chain, count

from .niche import Niche

import numpy as np
import pandas as pd
import torch


# import multiprocessing.pool
import multiprocessing as mp
from multiprocessing.pool import TimeoutError

class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc




class Algo:
    def __init__(self, 
                 evaluate_function,
                 learn_function,
                 environment_config,
                 agent_config,
                 save_path,
                 num_workers=10,
                 niche_num=10,
                 pair_agent_num=5,
                 reproduction_num_niche=10,
                 reproduction_num_agent=10,
                 admit_child_num=1,
                 reproduce_interval=2,
                 transfer_interval=20,
                 repro_env_interval=3,
                 repro_niche_threshold=8.0,
                 repro_agent_threshold=5.0,
                 mc_lower=1,
                 mc_upper=8,
                 novelty_knn=1,
                 checkpoint=None,
                 spawn=False,
                 timeout_develop=30,
                 timeout_evaluate=10,
                 timeout_transfer=120,
                 timeout_bootstrap=300):


        # def evaluate_function_wrapper(args):
        #     return evaluate_function(**args)
        
        # def learn_function_wrapper(args):
        #     return learn_function(**args)

        self.evaluate_function = evaluate_function
        self.learn_function = learn_function

        self.environment_config = environment_config
        self.agent_config = agent_config
        
        self.niche_num = niche_num
        self.pair_agent_num = pair_agent_num
        self.reproduction_num_niche = reproduction_num_niche
        self.reproduction_num_agent = reproduction_num_agent
        self.admit_child_num = admit_child_num
        self.transfer_limit = 10

        self.reproduce_interval = reproduce_interval
        self.transfer_interval = transfer_interval
        self.repro_env_interval = repro_env_interval
        
        self.repro_niche_threshold = repro_niche_threshold
        self.repro_agent_threshold = repro_agent_threshold

        self.mc_lower = mc_lower
        self.mc_upper = mc_upper
        
        self.novelty_knn = novelty_knn

        self.iteration = 0
        self.iteration_start_time = None

        self.niches = {}
        self.niches_archive = {}
        self.base_niche = None
        self.niche_indexer = count(0)

        self.live_agent_keys = set()

        if spawn:
            mp.set_start_method("spawn")
        else:
            mp.set_start_method("fork")
        self.num_workers = num_workers
        self.retry_limit = 10
        # self.paralell = NonDaemonPool(self.num_workers)
        self.paralell = mp.pool.Pool(self.num_workers)

        self.timeout_develop = timeout_develop
        self.timeout_evaluate = timeout_evaluate
        self.timeout_transfer = timeout_transfer
        self.timeout_bootstrap = timeout_bootstrap
       
        

        self.save_path = save_path
        self.niche_path = os.path.join(save_path, 'niche')
        os.makedirs(self.niche_path, exist_ok=True)
        self.agent_path = os.path.join(save_path, 'agent')
        os.makedirs(self.agent_path, exist_ok=True)

        self.develop_log = pd.DataFrame()
        self.log_path = os.path.join(self.save_path, 'log')
        os.makedirs(self.log_path, exist_ok=True)

        self.niche_history = {
            'filename': os.path.join(save_path, 'niches.csv'),
            'header': ['generated', 'key', 'parent']
        }
        self.agent_history = {
            'filename': os.path.join(save_path, 'agents.csv'),
            'header': ['generated', 'key', 'parent']
        }

        if checkpoint is None:
            self._init_log()
        else:
            with open(os.path.join(save_path, 'checkpoint', f'{checkpoint}', 'config.pickle'), 'rb') as f:
                self.environment_config, self.agent_config = pickle.load(f)

            with open(os.path.join(save_path, 'checkpoint', f'{checkpoint}', 'agents.pickle'), 'rb') as f:
                agents = pickle.load(f)

            with open(os.path.join(save_path, 'checkpoint', f'{checkpoint}', 'niches.pickle'), 'rb') as f:
                self.niches = pickle.load(f)

            with open(os.path.join(save_path, 'checkpoint', f'{checkpoint}', 'params.pickle'), 'rb') as f:
                self.iteration, self.niche_indexer = pickle.load(f)

            for niche_key,niche in self.niches.items():
                niche.agents = {agent_key: agents[agent_key] for agent_key in niche.agents}
                setattr(niche, 'results_tmp', {})

            self._make_base_niche()
            self.live_agent_keys = set(agents.keys())

            for agent_key,agent in agents.items():
                for key in list(agent.cores.keys()):
                    if key not in self.niches:
                        agent.drop_pair_niche(key)

            # self.end_run()
            # quit()

            # self._add_niches({}, {list(self.niches.keys())[0]: list(self.niches.values())[0]})


    def _init_log(self):
        with open(self.niche_history['filename'], 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.niche_history['header'])
            writer.writeheader()

        with open(self.agent_history['filename'], 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.agent_history['header'])
            writer.writeheader()


    def __write_log_niche(self, niche_key, niche):
        with open(self.niche_history['filename'], 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.niche_history['header'])
            items = {
                    'generated': self.iteration,
                    'key': niche_key,
                    'parent': niche.parent_key
            }
            writer.writerow(items)

    def __write_log_agent(self, agent_key, agent):
        with open(self.agent_history['filename'], 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.agent_history['header'])
            items = {
                    'generated': self.iteration,
                    'key': agent_key,
                    'parent': agent.parent_key,
            }
            writer.writerow(items)


    def _add_niches(self, new_niches, drop_niches):
        
        for niche_key,niche in new_niches.items():
            self.niches[niche_key] = niche
            # niche.admitted(self.environment_config)
            self.__write_log_niche(niche_key, niche)
        print('admitted niches: ' + ', '.join([f'{key: =6}' for key in new_niches.keys()]))

        for niche_key,niche in drop_niches.items():
            self.niches.pop(niche_key)
            for agent in niche.get_agents().values():
                agent.drop_pair_niche(niche_key)
            niche.archive()
            # self.niches_archive[niche_key] = niche
        print(f'archived niches: ' + ', '.join([f'{key: =6}' for key in drop_niches.keys()]))
        print()


    def _add_agents(self, new_agents):

        live_agents = dict(chain.from_iterable([niche.get_agents().items() for niche in self.niches.values()]))
        live_agent_keys = set(live_agents.keys())

        dropped = sorted(list(self.live_agent_keys - live_agent_keys))
        admitted = sorted(list(live_agent_keys - self.live_agent_keys))

        for agent_key in admitted:
            agent = live_agents[agent_key]
            self.__write_log_agent(agent_key, agent)
        
        print('admitted agents: ' + ' ,'.join(map(lambda z: f'{z: =6}', admitted)))
        print(' dropped agents: ' + ' ,'.join(map(lambda z: f'{z: =6}', dropped)))

        self.live_agent_keys = live_agent_keys

    def _make_base_niche(self):
        _, environment = self.environment_config.make_init(key=0)
        base_save_path = os.path.join(self.niche_path, '_base')
        self.base_niche = Niche(-1, -1, base_save_path)
        self.base_niche.set_environment(environment)

    def initialize(self):
        print('-----   Initialize   -----')
        start_time = time.time()

        _, environment = self.environment_config.make_init()

        base_save_path = os.path.join(self.niche_path, '_base')
        self.base_niche = Niche(-1, -1, base_save_path)
        self.base_niche.set_environment(environment)

        niche_key = next(self.niche_indexer)
        niche_path = os.path.join(self.niche_path, str(niche_key))
        niche = Niche(niche_key, -1, niche_path)
        niche.set_environment(environment)
        init_niches = {niche_key: niche}

        new_agents, initialize_log1 = self._reproduce_agent()
        initialize_log2 = self._evaluate(init_niches, {-1: self.base_niche})
        initialize_log3, initialize_log4 = self._transfer(init_niches, {-1: self.base_niche})

        self._add_niches(init_niches, {})
        self._add_agents(new_agents)

        self.base_niche.clear_agents()
        niche.end_iteration(self.iteration)

        end_time = time.time()
        print(f'elapsed time: {end_time - start_time: =.1f} sec')
        print('\n')

        initialize_log1.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_0_initialize1.csv'))
        initialize_log2.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_0_initialize2.csv'))
        initialize_log3.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_0_initialize3.csv'))
        initialize_log4.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_0_initialize4.csv'))


    def paralell_map(self, func, kwds, prefix='', timeout=None):
        results_niche = {}
        while kwds:
            bad_kwds = {}
            with tqdm(total=len(kwds), ncols=100, desc=prefix) as t:

                processes = {key: self.paralell.apply_async(func, kwds=kwds_) for key,kwds_ in kwds.items()}
                for key,process in processes.items():
                    success = False
                    try:
                        result = process.get(timeout=timeout)
                        success = True
                    except TimeoutError:
                        bad_kwds[key] = kwds[key]
                    except IndexError:
                        bad_kwds[key] = kwds[key]

                    if success:
                        niche_key, pair_key = key
                        if not niche_key in results_niche:
                            results_niche[niche_key] = {}
                        results_niche[niche_key][pair_key] = result

                        t.update(1)
                kwds = bad_kwds
        
        return results_niche

    
    def develop(self):
        print('-----   Develop   -----')

        kwds = {}
        for niche_key, niche in self.niches.items():
            kwds_ = niche.get_learn(self.environment_config, self.agent_config)
            kwds.update(kwds_)

        results = self.paralell_map(self.learn_function, kwds, prefix='learn   ', timeout=self.timeout_develop)

        for niche_key, niche in self.niches.items():
            niche.set_learn(self.iteration, results.get(niche_key, {}))

        for niche_key, niche in self.niches.items():
            niche.set_learn_result()
        

        kwds = {}
        for niche_key, niche in self.niches.items():
            kwds_ = niche.get_evaluate(self.environment_config, self.agent_config)
            kwds.update(kwds_)

        results = self.paralell_map(self.evaluate_function, kwds, prefix='evaluate', timeout=self.timeout_evaluate)

        for niche_key, niche in self.niches.items():
            niche.set_evaluate(self.iteration, results.get(niche_key, {}))


        print('   niche     ' + ('   ').join([' agent| reward']*self.pair_agent_num) + '        best     ')
        print('             ' + ('   ').join(['==============']*self.pair_agent_num) + '   ==============')
        for niche_key, niche in self.niches.items():
            rewards = niche.get_rewards()
            best_agent_key, best_reward = niche.get_best_reward()
            print(
                f'  {niche_key: =6} :   ' + \
                '   '.join([f'{agent_key: =6}| {reward: =+6.2f}' for (_,agent_key),reward in rewards.items()] + [' '*14]*(self.pair_agent_num - len(rewards))) + \
                f'   {best_agent_key: =6}| {best_reward: =+6.2f}'
            )

            for pair_key, reward in rewards.items():
                pair_str = '{0}-{1}'.format(*pair_key)
                self.develop_log.loc[self.iteration, pair_str] = reward
        print()



    def _evaluate(self, recievers, senders):

        log = pd.DataFrame(columns=list(map(lambda z: '{0}-{1}'.format(*z), (chain.from_iterable([niche.get_pair_keys() for niche in senders.values()])))))

        kwds = {}
        for reciever_key,reciever_niche in recievers.items():
            senders_ = {sender_key: sender_niche for sender_key,sender_niche in senders.items() if sender_key != reciever_key}
            kwds_ = reciever_niche.get_evaluate(self.environment_config, self.agent_config, niches=senders_)
            kwds.update(kwds_)

        results = self.paralell_map(self.evaluate_function, kwds, prefix='evaluate', timeout=self.timeout_evaluate)

        for reciever_key, reciever_niche in recievers.items():
            senders_ = {sender_key: sender_niche for sender_key,sender_niche in senders.items() if sender_key != reciever_key}
            reciever_niche.set_evaluate(self.iteration, results.get(reciever_key, {}), niches=senders_)

            pair_keys = list(chain.from_iterable([niche.get_pair_keys() for niche in senders_.values()]))
            rewards = reciever_niche.get_rewards(pair_keys=pair_keys)
            for pair_key, reward in rewards.items():
                pair_str = '{0}-{1}'.format(*pair_key)
                log.loc[reciever_key, pair_str] = reward

        return log

    def _transfer(self, recievers, senders):

        log1 = pd.DataFrame(columns=list(map(lambda z: '{0}-{1}'.format(*z), (chain.from_iterable([niche.get_pair_keys() for niche in senders.values()])))))
        
        kwds = {}
        for niche_key, niche in recievers.items():
            senders_ = {sender_key: sender_niche for sender_key,sender_niche in senders.items() if sender_key != niche_key}
            pass_agent_pair_keys = niche.get_pass_pair_keys(senders_, self.transfer_limit)
            kwds_ = niche.get_learn(self.environment_config, self.agent_config, niches=senders_, pair_keys=pass_agent_pair_keys, transfer=True)
            kwds.update(kwds_)

        results = self.paralell_map(self.learn_function, kwds, prefix='learn   ', timeout=self.timeout_transfer)

        for niche_key, niche in recievers.items():
            senders_ = {sender_key: sender_niche for sender_key,sender_niche in senders.items() if sender_key != niche_key}
            niche.set_learn(self.iteration, results.get(niche_key, {}), niches=senders_)

            rewards = niche.get_imigrant_rewards()
            for pair_key, reward in rewards.items():
                pair_str = '{0}-{1}'.format(*pair_key)
                log1.loc[niche_key, pair_str] = reward


        log2 = pd.DataFrame(columns=['dropped', 'accepted'])
        width1 = 6 * self.pair_agent_num + 2 * (self.pair_agent_num - 1)
        pad1 = ' ' * ((width1 - 8) // 2)
        width2 = 13 * self.pair_agent_num + 2 * (self.pair_agent_num - 1)
        pad2 = ' ' * ((width2 - 8) // 2)
        print('   niche     ' + pad1 + ' dropped' + pad1 + '   ' + pad2 + 'accepted')
        print('             ' + '=' * width1 + '   ' + '=' * width2)
        for niche_key, niche in recievers.items():
            baselines = niche.get_baselines()
            drop_keys, accept_pairs = niche.transferred(self.iteration, self.pair_agent_num, self.agent_config)

            print(
                f'  {niche_key: 6} :   ' + \
                ', '.join(map(lambda z: f'{z}', drop_keys)).rjust(width1) + '   ' + \
                ', '.join(map(lambda z: f'{z[1]}@{z[0]}', accept_pairs)).rjust(width2)
            )

            log2.loc[niche_key] = [','.join(map(str, drop_keys)), ','.join(map(lambda z: f'({z[0]},{z[1]})', accept_pairs))]
        
        print()

        return log1, log2


    def transfer(self):
        print('-----   Transfer   -----')

        recievers = self.niches
        senders = self.niches

        transfer_log1 = self._evaluate(recievers, senders)
        transfer_log2,transfer_log3 = self._transfer(recievers, senders)

        self._add_agents({})

        print()

        transfer_log1.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_3_tranfer1.csv'))
        transfer_log2.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_3_tranfer2.csv'))
        transfer_log3.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_3_tranfer3.csv'))



    def __pass_mc(self, reward):
        return reward > self.mc_lower and reward < self.mc_upper

    def __get_novelty_scores_niche(self, child_niches):
        
        pair_keys = list(chain.from_iterable([niche.get_pair_keys() for niche in self.niches.values()]))
        live_rewards = np.vstack([list(niche.get_rewards(pair_keys).values()) for niche in self.niches.values()])

        novelty_scores = {}
        for key,child in child_niches.items():
            child_rewards = np.array(list(child.get_rewards(pair_keys).values()))
            distances = np.linalg.norm(np.expand_dims(child_rewards, axis=0) - live_rewards, axis=1)
            knn_distances = np.sort(distances)[:self.novelty_knn]
            novelty_scores[key] = knn_distances.mean()

        return novelty_scores

    def _reproduce_niches(self, do=True):
        if not do:
            return {}, {}, (pd.DataFrame(), pd.DataFrame())

        print('... reproduce niches ...')

        parent_niches = [(niche_key,niche) for niche_key,niche in self.niches.items() if niche.check_parent(self.repro_niche_threshold)]

        if len(parent_niches) == 0:
            print('no parent niches')
            return {}, {}, (pd.DataFrame(), pd.DataFrame())

        print('parent niches: ' + ', '.join(map(lambda z: f'{z[0]: =6}', parent_niches)))
        
        child_candidates = {}
        for _ in range(self.reproduction_num_niche):
            parent_key, parent_niche = random.choice(parent_niches)
            child_key = next(self.niche_indexer)
            child_niche = parent_niche.reproduce(child_key, self.niche_path, self.environment_config)
            child_candidates[child_key] = child_niche
            
        recievers = dict(list(child_candidates.items()) + list(self.niches.items()))
        log2 = self._evaluate(recievers, self.niches)

        novelty_scores = self.__get_novelty_scores_niche(child_candidates)


        log1 = pd.DataFrame(columns=['parent', 'best_agent_key', 'best_reward', 'elites_reward', 'mc', 'novelty'])
        child_niches = []
        print('  child     parent        best        elites    mc     novelty')
        print('            ======   ==============   ======   ====   ========')
        for child_key,child_niche in child_candidates.items():

            best_agent_key, best_reward = child_niche.get_best_reward()
            elites_reward = child_niche.get_elites_reward(self.pair_agent_num)
            novelty_score = novelty_scores[child_key]

            print(f' {child_key: =6} :   {child_niche.parent_key: =6}   {best_agent_key: =6}| {best_reward: =+6.2f}   {elites_reward: =6.2f}', end='   ')

            log1.loc[child_key, ['parent', 'best_agent_key', 'best_reward', 'elites_reward', 'novelty']] = [child_niche.parent_key, best_agent_key, best_reward, elites_reward, novelty_score]

            if self.__pass_mc(elites_reward):
                print(f'pass   {novelty_score: =8.2f}')
                child_niches.append((child_key, child_niche, novelty_score))
                log1.loc[child_key, 'mc'] = 1
            else:
                print(f'fail')
                log1.loc[child_key, 'mc'] = 0

        child_niches = sorted(child_niches, key=lambda x: x[2], reverse=True)
        new_niches = {key: niche for key,niche,_ in child_niches[:self.admit_child_num]}

        drop_num =  max(len(self.niches) + self.admit_child_num - self.niche_num, 0)
        drop_niches = dict(sorted(self.niches.items(), key=lambda z: z[0])[:drop_num])
        
        print()

        return new_niches, drop_niches, (log1, log2)


    def _bootstrap(self, agents):

        log = pd.DataFrame(columns=['bootstrap', 'reward'])

        self.base_niche.set_agents(agents)

        kwds = self.base_niche.get_learn(self.environment_config, self.agent_config, bootstrap=True)
        results = self.paralell_map(self.learn_function, kwds, prefix='laern   ', timeout=self.timeout_bootstrap)
        self.base_niche.set_learn(self.iteration, results[-1])

        self.base_niche.set_learn_result()

        kwds = self.base_niche.get_evaluate(self.environment_config, self.agent_config)
        results = self.paralell_map(self.evaluate_function, kwds, prefix='evaluate', timeout=self.timeout_evaluate)
        self.base_niche.set_evaluate(self.iteration, results[-1])

        rewards = self.base_niche.get_rewards()
        for pair_key,reward in rewards.items():
            niche_key,agent_key = pair_key
            log.loc[agent_key] = [niche_key, reward]

        self.base_niche.end_iteration(self.iteration)

        return rewards, log

    def _reproduce_agent(self, do=True):
        if not do:
            return {}, pd.DataFrame()

        print('... reproduce agents ...')

        parent_agents = sorted(list(set(list(chain.from_iterable([list(niche.get_parent_agent(self.repro_agent_threshold).items()) for niche in self.niches.values()])))))

        print('parent agents: ' + ', '.join(map(lambda z: f'{z[0]: =6}', parent_agents)))

        child_agents = {}
        log = pd.DataFrame(columns=['parent'])
        for _ in range(self.reproduction_num_agent):
            if len(parent_agents) == 0:
                parent_key = -1
                child_key, child_agent = self.agent_config.make_init()
            else:
                parent_key, parent_agent = random.choice(parent_agents)
                child_key, child_agent = self.agent_config.reproduce(parent=parent_agent)

            child_agents[child_key] = child_agent

            log.loc[child_key] = [parent_key]

        bootstrap_rewards, log_ = self._bootstrap(child_agents)
        log = pd.concat([log, log_], axis=1)

        print('   child     parent   bootstrap niche   bootstrap reward')
        print('             ======   ===============   ================')
        for child_key,child_agent in child_agents.items():
            print(f'  {child_key: 6} :   {child_agent.parent_key: =6}                -1   {bootstrap_rewards[(-1,child_key)]: =+16.2f}')

        print()

        return child_agents, log

    def reproduce(self, do_environment=True, do_agent=True):
        print('-----   Reproduce   -----')
        
        new_niches, drop_niches, (reproduce_log1, reproduce_log2) = self._reproduce_niches(do=do_environment)

        new_agents, reproduce_log3 = self._reproduce_agent(do=do_agent)

        recievers = dict(list({niche_key: niche for niche_key,niche in self.niches.items() if niche_key not in drop_niches}.items()) + list(new_niches.items()))
        senders = dict(list(self.niches.items()) + [(-1, self.base_niche)])

        reproduce_log4 = self._evaluate(recievers, senders)
        reproduce_log5, reproduce_log6 = self._transfer(recievers, senders)

        self._add_niches(new_niches, drop_niches)
        self._add_agents(new_agents)

        self.base_niche.clear_agents()

        print()


        reproduce_log1.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce1.csv'))
        reproduce_log2.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce2.csv'))
        reproduce_log3.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce3.csv'))
        reproduce_log4.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce4.csv'))
        reproduce_log5.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce5.csv'))
        reproduce_log6.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_4_reproduce6.csv'))


    def start_iteration(self):
        self.iteration += 1
        print(f'********************  ITERATION {self.iteration: =6}   ********************')
        self.iteration_start_time = time.time()
        print()

    
    def end_iteration(self):
        for niche in self.niches.values():
            niche.end_iteration(self.iteration)

        iteration_end_time = time.time()
        print(f'elapsed time: {iteration_end_time - self.iteration_start_time: =.1f} sec')
        print('\n')


    def end_run(self):
        for niche_key,niche in self.niches.items():
            niche.archive()

    
    def run(self, iterations):
        
        while self.iteration < iterations:
            self.start_iteration()

            self.develop()

            if self.iteration % self.transfer_interval == 0:

                self.develop_log.to_csv(os.path.join(self.log_path, f'{self.iteration:#=6}_1_develop.csv'))
                self.develop_log = pd.DataFrame()

                for niche in self.niches.values():
                    niche.save_cores(self.iteration)
                
                if self.iteration % (self.reproduce_interval * self.transfer_interval) == 0:
                    do_agent = True
                    do_environment = self.iteration % (self.reproduce_interval * self.transfer_interval * self.repro_env_interval) == 0
                    self.reproduce(do_environment=do_environment, do_agent=do_agent)
                
                elif len(self.niches) > 1:
                    self.transfer()


            self.end_iteration()

            if self.iteration % self.transfer_interval == 0:
                self.checkpoint()

        
        self.end_run()


    def checkpoint(self):
        tmp_path = os.path.join(self.save_path, 'checkpoint', f'{self.iteration}')
        os.makedirs(tmp_path, exist_ok=True)

        agents = dict(chain.from_iterable([niche.get_agents().items() for niche in self.niches.values()]))

        # agents = {}
        for niche_key,niche in self.niches.items():
            # history = niche.history
            # history.to_csv(os.path.join(tmp_path, f'history_{niche.key}.csv'))

            # agents_ = niche.get_agents()
            # for agent_key,agent in agents_.items():
            #     core = agent.get_core(niche_key)
            #     torch.save([core['policy_params'], core['obs_rms'], core['opt_params']], os.path.join(tmp_path, f'{agent_key}_{niche_key}.pt'))

            #     agents[agent_key] = agent

            niche.agents = niche.get_agent_keys()
            niche.processes = {}
            # niche.history = None

        # for agent_key,agents in agents.items():
        #     agent.cores = {}
        #     agent.pair_niche_key = []

        with open(os.path.join(tmp_path, 'niches.pickle'), 'wb') as f:
            pickle.dump(self.niches, f)

        with open(os.path.join(tmp_path, 'agents.pickle'), 'wb') as f:
            pickle.dump(agents, f)

        with open(os.path.join(tmp_path, 'config.pickle'), 'wb') as f:
            pickle.dump([self.environment_config, self.agent_config], f)

        with open(os.path.join(tmp_path, 'params.pickle'), 'wb') as f:
            pickle.dump([self.iteration, self.niche_indexer], f)

        for niche_key,niche in self.niches.items():
            niche.agents = {agent_key: agents[agent_key] for agent_key in niche.agents}

        



        
