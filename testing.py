import os
import sys
import argparse
#import torch#

#env_path = os.path.join(os.path.dirname(__file__), '..')
#if env_path not in sys.path:
#    sys.path.append(env_path)
    
from test.evaluation import Running
#torch.cuda.set_device(1)#


def Run(test_name, dataset_name='pascal'):
    return Running(test_name, dataset_name)


def main():
    parser = argparse.ArgumentParser(description='Run sodgan method on dataset.')
    parser.add_argument('test_name', type=str, help='Name of testing method.')
    parser.add_argument('--dataset', type=str, default='pascal', help='Name of dataset (ecssd, hkuis, pascal, dutomron, dutste).')

    args = parser.parse_args()

    runner = Run(args.test_name, args.dataset)
    runner.infer()


if __name__ == '__main__':
    main()
