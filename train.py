import parser
import loaders
from network import Network

if __name__ == '__main__':
	arg = parser.parser.parse_args()
	print(arg)
	train_loader, valid_loader = loaders.load(arg.data_directory)
	n = Network(arg.arch, arg.hidden_units, arg.learning_rate)
	n.train(arg.gpu, arg.epochs, train_loader, valid_loader)
