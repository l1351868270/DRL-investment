
import argparse
import importlib
import logging
import os
from datetime import datetime

from stable_baselines3.common.env_checker import check_env

from drl_investment.data.tdx import unpack_data

LOG = logging.getLogger(__name__)


def valid_date(s: str) -> None:
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except:
        raise argparse.ArgumentTypeError('The date format is YYYY-MM-DD. Please use --help for help')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=f'{os.path.join(os.path.dirname(__file__), "../..", "tests/assets/sh000001.day")}')
    parser.add_argument('--begin_date', type=valid_date, default='2006-01-01', help='The date format is YYYY-MM-DD')
    parser.add_argument('--env', type=str, default='drl_investment.envs.stocks_v4.StocksEnvV4')
    parser.add_argument('--policy', choices=('MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'), default='MlpPolicy')
    parser.add_argument('--algo', type=str, default='PPO', help='See stable_baselines3 support algos')
    parser.add_argument('--timesteps', type=int, default=1500000, help='Every loop run n timesteps')
    parser.add_argument('--loop', type=int, default=1000, help='Run n loops, every loop run m timesteps')
    parser.add_argument('--log', choices=('debug', 'info', 'warning', 'error', 'fatal', 'critical'), default='debug')
    
    args = parser.parse_args()
    return args


def set_log(level):
    handler = logging.StreamHandler()
    LOG.addHandler(handler)
    LOG.setLevel(eval(f'logging.{level.upper()}'))


def make_env(args):
    df = unpack_data(args.data_path)[args.begin_date:]
    name, package = args.env.rsplit('.', 2)
    cenv = importlib.import_module(name, package=package)
    env_config = {
        'data': df
    }
    env = cenv(config=env_config)
    check_env(env)
    return env


def train(args, env):
    tensorboard_log = os.path.expanduser(os.path.join('~/sb3_results', args.env, args.algo))
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    algo = importlib.import_module('stable_baselines3', package=args.algo)
    # from stable_baselines3 import PPO
    # PPO('MlpPolicy', env, verbose=2, tensorboard_log=tensorboard_log)
    model = algo(args.policy, env, verbose=2, tensorboard_log=tensorboard_log)
    LOG.info(model.policy)

    begin_time = datetime.now()
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(model.logger.dir, f'model'))
    LOG.info(f'The begin loop cost {datetime.now() - begin_time}s')
    for i in range(1, args.loops):
        begin_time = datetime.datetime.now()
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        model.save(os.path.join(model.logger.dir, f'model_{i*args.timesteps}'))
        LOG.info(f'The {i} loop ({i*args.timesteps}-{(i+1)*args.timesteps-1} steps) cost {datetime.datetime.now() - begin_time}s')


def main():
    args = get_args()
    set_log(args.log)
    env = make_env(args)
    train(args, env)


if __name__ == '__main__':
    # python -m
    main()

