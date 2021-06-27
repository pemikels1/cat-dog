import argparse
import os


def park_parse_args():

    parser = argparse.ArgumentParser(description='Cat Dog Classifier')

    parser.add_argument('--batch_size', default=16, help='Number of observations sent at a time')
    parser.add_argument('--epochs', default=15, help='Number of passes through dataset')
    parser.add_argument('--shuffle', default=True, help='Shuffle data after every epoch')
    parser.add_argument('--lr', default=0.5, help='Initial learning rate')
    parser.add_argument('--step_size', default=2, help='Interval for LR rate decay')
    parser.add_argument('--gamma', default=0.7, help='LR rate decay factor')
    parser.add_argument('--train_percentage', default=0.8, help='Percentage of data in training set')
    parser.add_argument('--seed', default=1, help='Random seed')
    parser.add_argument('--data', default=os.environ['DATA_PATH'], help='Path to data folder')
    parser.add_argument('--checkpoint', default=os.environ['CHECKPOINTS_PATH'], help='Path to checkpoints folder')

    args = parser.parse_args()

    return args
