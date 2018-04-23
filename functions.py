from PIL import Image
import numpy as np

def process_image(filename):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
	im = Image.open(filename)
	im = im.resize((256,256))
	im = im.crop((16,16,240,240))
	im = np.array(im) / 255
    
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	im = (im - mean) / std
    
	return im.transpose(2,0,1)
