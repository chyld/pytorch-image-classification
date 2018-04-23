import parser
import loaders
from model import Model

if __name__ == '__main__':
	print(parser.parser.parse_args())
	train_loader, valid_loader, num_train_images, num_valid_images, train_dataset = loaders.load(parser.parser.parse_args().data_directory)
	m = Model(parser.parser.parse_args().arch, parser.parser.parse_args().hidden_units, parser.parser.parse_args().learning_rate)
	m.create()
	m.train(parser.parser.parse_args().gpu, parser.parser.parse_args().epochs, parser.parser.parse_args().save_dir, train_loader, valid_loader, num_train_images, num_valid_images)
	m.save(parser.parser.parse_args().save_dir, train_dataset)
