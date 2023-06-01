import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Evogym POET experiment'
    )

    parser.add_argument(
        '-n', '--name',
        default='default', type=str,
        help='experiment name (default: default)'
    )
    parser.add_argument(
        '-t', '--task',
        default='Parkour-v0', type=str,
        help='task name (default: Parkour-v0)'
    )

    parser.add_argument(
        '-i', '--iteration',
        default=10000, type=int,
        help='iteration of poet (default: 3000)'
    )
    parser.add_argument(
        '-n-num', '--niche-num',
        default=16, type=int,
        help='limit for pair to hold (default: 10)'
    )
    parser.add_argument(
        '-a-num', '--pair-agent-num',
        default=8, type=int,
        help='limit for pair agent num to hold each niche (default: 5)'
    )
    parser.add_argument(
        '-r-num-n', '--reproduce-num-niche',
        default=16, type=int,
        help='number of reproduce at once (default: 10)'
    )
    parser.add_argument(
        '-r-num-a', '--reproduce-num-agent',
        default=10, type=int,
        help='number of reproduce at once (default: 10)'
    )
    parser.add_argument(
        '-ac-num', '--admit-child-num',
        default=1, type=int,
        help='limit for admission of child at once (default: 1)'
    )
    parser.add_argument(
        '-r-iv', '--reproduce-interval',
        default=2, type=int,
        help='reproduce interval (default: 4)'
    )
    parser.add_argument(
        '-t-iv', '--transfer-interval',
        default=10, type=int,
        help='transfer interval (default: 10)'
    )
    parser.add_argument(
        '-r-n-th', '--reproduce-niche-threshold',
        default=0.8, type=float,
        help='ratio to maximum reward. used for threshold of reward to reproduce new niche (default: 0.8)'
    )
    parser.add_argument(
        '-r-a-th', '--reproduce-agent-threshold',
        default=0.5, type=float,
        help='ratio to maximum reward. used for threshold of reward to reproduce new agent (default: 0.5)'
    )
    parser.add_argument(
        '-mc-l', '--mc-lower',
        default=0.1, type=float,
        help='ratio to maximum reward. used for lower minimal criterion. (default: 0.1)'
    )
    parser.add_argument(
        '-mc-u', '--mc-upper',
        default=0.8, type=float,
        help='ratio to maximum reward. used for upper minimal criterion. (default: 0.8)'
    )

    parser.add_argument(
        '-w', '--width',
        default=100, type=int,
        help='width of evogym terrain (default: 100)'
    )
    parser.add_argument(
        '-fp', '--first-platform',
        default=10, type=int,
        help='first platform width of evogym terrain (default: 10)'
    )

    parser.add_argument(
        '-si-ppo', '--steps-per-iteration',
        default=4, type=int,
        help='number of steps per iteration of poet (default: 4)'
    )
    parser.add_argument(
        '-sb-ppo', '--steps-bootstrap',
        default=4*32, type=int,
        help=''
    )
    parser.add_argument(
        '-st-ppo', '--steps-transfer',
        default=4*8, type=int,
        help=''
    )
    parser.add_argument(
        '-lr-ppo', '--learning-rate',
        default=2.5e-4, type=float,
        help='learning rate (default: 2.5e-4)'
    )
    parser.add_argument(
        '-e-ppo', '--epoch',
        default=4, type=int,
        help='number of ppo epochs per 1 step of ppo (default: 4)'
    )
    parser.add_argument(
        '-b-ppo','--num-mini-batch',
        default=4, type=int,
        help='number of batches for ppo (default: 4)'
    )
    parser.add_argument(
        '-c-ppo', '--clip-range',
        default=0.1, type=float,
        help='ppo clip parameter (default: 0.1)'
    )
    parser.add_argument(
        '-s-ppo', '--steps',
        default=128, type=int,
        help='num steps to use in PPO (default: 128)'
    )
    parser.add_argument(
        '-p-ppo', '--num-processes',
        default=4, type=int,
        help='number of paralell environment processes for ppo (default: 4)'
    )
    parser.add_argument(
        '-std-ppo', '--init-log-std',
        default=0.3, type=float,
        help='initial log std of action distribution (default: 0.0)'
    )

    parser.add_argument(
        '--checkpoint',
        type=int,
        help=''
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=10, type=int,
        help='how many training CPU processes to use (default: 10)'
    )

    parser.add_argument(
        '--spawn',
        action='store_true', default=False,
        help=''
    )

    parser.add_argument(
        '-to-d', '--timeout-develop',
        default=60, type=float,
        help=''
    )
    parser.add_argument(
        '-to-e', '--timeout-evaluate',
        default=20, type=float,
        help=''
    )
    parser.add_argument(
        '-to-t', '--timeout-transfer',
        default=180, type=float,
        help=''
    )
    parser.add_argument(
        '-to-b', '--timeout-bootstrap',
        default=480, type=float,
        help=''
    )
    args = parser.parse_args()

    assert args.mc_lower < args.mc_upper, 'argument error: mc_lower < mc_upper'

    return args


def get_figure_args():
    parser = argparse.ArgumentParser(
        description='make robot figures'
    )

    parser.add_argument(
        'name',
        type=str,
        help='name of experiment for making figures'
    )
    parser.add_argument(
        '-s', '--specified',
        type=int,
        help='input id, make figure for the only specified niche (usage: "-s {id}")'
    )

    parser.add_argument(
        '-st', '--save-type',
        type=str, default='gif',
        help='file type (default: gif, choose from [gif, jpg])'
    )

    parser.add_argument(
        '-tr', '--track-robot',
        action='store_true', default=False,
        help='track robot with camera in gif'
    )

    parser.add_argument(
        '-i', '--interval',
        type=str, default='timestep',
        help='in case of save type is jpg, type of interval for robot drawing (default: timestep, choose from [timestep, distance])'
    )
    parser.add_argument(
        '-rs', '--resolution-scale',
        type=float, default=32.0,
        help='resolution scale. <br> when output monochrome image, try this argument change. (default: 32.0)'
    )
    parser.add_argument(
        '--start-timestep',
        type=int, default=0,
        help='start timestep of render (default: 0)'
    )
    parser.add_argument(
        '-ti', '--timestep-interval',
        type=int, default=80,
        help='timestep interval for robot drawing (default: 80, if interval is hybrid, it should be about 40)'
    )
    parser.add_argument(
        '-b', '--blur',
        type=int, default=0,
        help='in case of jpg, timesteps for rendering motion blur (default: 0, 0 means no blur)'
    )
    parser.add_argument(
        '-bt', '--blur-temperature',
        type=float, default=0.6,
        help='blur temperature (default: 0.6, up to 1.0)'
    )
    parser.add_argument(
        '-di', '--distance-interval',
        type=float, default=0.8,
        help='distance interval for robot drawing'
    )
    parser.add_argument(
        '--display-timestep',
        action='store_true', default=False,
        help='display timestep above robot'
    )
    parser.add_argument(
        '--draw-trajectory',
        action='store_true', default=False,
        help='draw robot trajectory as line'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=1, type=int,
        help='number of parallel making processes (default: 1)'
    )
    parser.add_argument(
        '--not-overwrite',
        action='store_true', default=False,
        help='skip process if already figure exists (default: False)'
    )
    parser.add_argument(
        '--no-multi',
        action='store_true', default=False,
        help='do without using multiprocessing. if error occur, try this option. (default: False)'
    )

    args = parser.parse_args()

    assert args.name is not None, 'argumented error: input "{experiment name}"'

    return args