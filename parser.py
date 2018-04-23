import argparse

parser = argparse.ArgumentParser(description='Training a Pytorch model')

parser.add_argument('data_directory', action='store')
parser.add_argument('--arch', action='store', default='resnet101')
parser.add_argument('--hidden_units', action='store', type=int, default=512)
parser.add_argument('--learning_rate', action='store', type=float, default=0.00001)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--epochs', action='store', type=int, default=25)
parser.add_argument('--save_dir', action='store', default='checkpoints')
