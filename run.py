
import sys
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)
import neat_cppn
from experiment_utils import initialize_experiment

from algo.algo import Algo
from algo.agent import AgentConfig, evaluate, learn
from algo.environment import EnvrionmentConfig


import custom_envs.parkour

from arguments.main import get_args


def main():

    args = get_args()
    save_path = os.path.join(CURR_DIR, 'out', 'main', args.name)

    if args.checkpoint is None:
        initialize_experiment(args.name, save_path, args)

    terrain_config_file = os.path.join(CURR_DIR, 'config', 'terrain_cppn.cfg')
    terrain_cppn_config = neat_cppn.make_config(terrain_config_file)
    terrain_cppn_config_file = os.path.join(save_path, 'evogym_terrain.cfg')
    terrain_cppn_config.save(terrain_cppn_config_file)

    env_config = EnvrionmentConfig(
        terrain_cppn_config,
        env_id=args.task,
        max_width=args.width,
        first_platform=args.first_platform)


    robot_config_file = os.path.join(CURR_DIR, 'config', 'robot_cppn.cfg')
    robot_cppn_config = neat_cppn.make_config(robot_config_file)
    robot_cppn_config_file = os.path.join(save_path, 'evogym_terrain.cfg')
    robot_cppn_config.save(robot_cppn_config_file)

    agent_path = os.path.join(save_path, 'agent')

    agent_config = AgentConfig(
        (5,5),
        robot_cppn_config,
        agent_path,
        steps_per_iteration=args.steps_per_iteration,
        steps_bootstrap=args.steps_bootstrap,
        steps_transfer=args.steps_transfer,
        clip_range=args.clip_range,
        epochs=args.epoch,
        num_mini_batch=args.num_mini_batch,
        steps=args.steps,
        num_processes=args.num_processes,
        learning_rate=args.learning_rate,
        init_log_std=args.init_log_std,
        max_steps=args.steps_per_iteration*args.reproduce_interval*20)

    if args.task=='Parkour-v1':
        maximum_reward = args.width/10 + 10
    else:
        maximum_reward = args.width/10

    poet_pop = Algo(
        evaluate,
        learn,
        env_config,
        agent_config,
        save_path,
        num_workers=args.num_cores,
        niche_num=args.niche_num,
        pair_agent_num=args.pair_agent_num,

        reproduction_num_niche=args.reproduce_num_niche,
        reproduction_num_agent=args.reproduce_num_agent,
        admit_child_num=args.admit_child_num,
        reproduce_interval=args.reproduce_interval,
        transfer_interval=args.transfer_interval,
        repro_niche_threshold=maximum_reward*args.reproduce_niche_threshold,
        repro_agent_threshold=maximum_reward*args.reproduce_agent_threshold,
        mc_lower=maximum_reward*args.mc_lower,
        mc_upper=maximum_reward*args.mc_upper,
        novelty_knn=1,
        checkpoint=args.checkpoint,
        spawn=args.spawn,
        timeout_develop=args.timeout_develop,
        timeout_evaluate=args.timeout_evaluate,
        timeout_transfer=args.timeout_transfer,
        timeout_bootstrap=args.timeout_bootstrap)

    if args.checkpoint is None:
        poet_pop.initialize()

    poet_pop.run(args.iteration)


if __name__=='__main__':
    main()