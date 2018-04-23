import argparse

parser = argparse.ArgumentParser(description='Prediction from a Pytorch model')

parser.add_argument('input', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--top_k', action='store', type=int, default=3)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)
