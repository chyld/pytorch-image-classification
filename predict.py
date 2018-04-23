import preparser as parser
from functions import process_image

if __name__ == '__main__':
	print(parser.parser.parse_args())
	print(process_image(parser.parser.parse_args().input))
