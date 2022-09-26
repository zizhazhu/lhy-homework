import argparse


def common_parser():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--trail_id', type=str, default='test')
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--output_dir', type=str, default='./result')
    parser.add_argument('mode', type=str)
    return parser
