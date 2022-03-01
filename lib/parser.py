# argument parser
import argparse
import numpy as np

def parse_args():
    '''parse input arguments'''
    parser = argparse.ArgumentParser(
        description='If we could using the training data many times')

    parser.add_argument('--learning-rate', default=0.01, \
        type=float)
    parser.add_argument('--skip-frame', default=3, type=int)
    parser.add_argument('--buffer-size', default=2000, type=int)
    parser.add_argument('--task', default='ALE/Pong-v5')
    parser.add_argument('--obs-merge', default=3, type=int, 
        help='merge the neibor obs')
    parser.add_argument('--frame', default=84, type=int)
    parser.add_argument('--train-parallel', default=8, \
        type=int, help='the nunmber of the training prarallel')
    parser.add_argument('--test-parallel', default=8, type=int, 
        help='the number of the model used to test the perfermace of the model parallely')
    parser.add_argument('--discount-rate', default=0.99, type=float, 
        help='the discount rate of the training process')
    parser.add_argument('--sync-steps', default=1000, type=int, 
        help='the step to sync the trained net to target net')
    parser.add_argument('--n-step', default=3, type=int, 
        help='the step used to estimate the value of the action')
    parser.add_argument('--test-per-eps', default=40, type=int, 
        help='test the policy of the agent')
    parser.add_argument('--is-shallow', action='store_false')
    parser.add_argument('--origin', action='store_true', 
        help='whether use origin dqn')
    parser.add_argument('--logdir', '-p',  default='log', 
        help='the path to save the weights, logs')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--max-step', default=300, type=int)

    args = parser.parse_args()
    return args
