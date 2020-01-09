import os
import sys
import argparse
import importlib
import multiprocessing
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import train.admin.settings as ws_settings

def run_training(train_name, cudnn_benchmark=True):
    
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.manual_seed(2018)
    
    print('Training: {}'.format(train_name))
    
    settings = ws_settings.Settings()
    settings.script_name = train_name
    settings.project_path = 'train/{}'.format(train_name)

    expr_module = importlib.import_module('train.train_settings.{}'.format(train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)
    
def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()

    run_training(args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
