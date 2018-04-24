import preparser as parser
from functions import process_image, hydrate, cat_to_name

if __name__ == '__main__':
	arg = parser.parser.parse_args()
	print(arg)
	image = process_image(arg.input)
	network = hydrate(arg.checkpoint)
	names = cat_to_name(arg.category_names)
	prediction = network.predict(image, arg.gpu, arg.top_k, names)
	print(prediction)
