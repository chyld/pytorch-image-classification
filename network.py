import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

class Network:
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
	def __init__(self, arch, units, lr):
		# currently works with resnet, densenet & inception models
		self.model = getattr(models, arch)(pretrained=True)
		class_layer_name = 'classifier' if 'classifier' in dir(self.model) else 'fc'
		in_size = getattr(self.model, class_layer_name).in_features
		out_size = 102

        # freeze parameters
		for param in self.model.parameters():
			param.requires_grad = False

        # create output classification layer
		output_layer = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_size, units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(units, out_size)),
                          ('relu', nn.ReLU()),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        # replace old classification layer with custom one
		setattr(self.model, class_layer_name, output_layer)
        
        # loss and optimizer functions
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(getattr(self.model, class_layer_name).parameters(), lr=lr)
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
	def train(self, gpu, epochs, train_loader, valid_loader):
		cuda = torch.cuda.is_available()

		if cuda and gpu:
			self.model.cuda()
		else:
			self.model.cpu()
        
		loss_per_x_batches = 0
		print_every_x_batches = 50

		self.model.train();
        
		for epoch in range(epochs):
			for ii, (inputs, labels) in enumerate(train_loader):
                ### -------------------------------------------------------------------------------- ###
				inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

				if cuda and gpu:
					inputs, labels = inputs.cuda(), labels.cuda()

				outputs = self.model.forward(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				loss_per_x_batches += loss.data[0]
                ### -------------------------------------------------------------------------------- ###
				if ii % print_every_x_batches == 0:
					self.model.eval()
					validation_loss, total_score = 0, 0
            
					for _, (inputs, labels) in enumerate(valid_loader):                
						inputs, labels = Variable(inputs, requires_grad=False, volatile=True), Variable(labels)

						if cuda and gpu:
							inputs, labels = inputs.cuda(), labels.cuda()

						outputs = self.model.forward(inputs)
						loss = self.criterion(outputs, labels)

						validation_loss += loss.data[0]
						probabilities = torch.exp(outputs).data
						total_score += (labels.data == probabilities.max(1)[1]).sum()
                  
                  
					print(
						"epoch: {} batch: {}".format(epoch, ii),
						"train loss: {:.3f}".format(loss_per_x_batches),
						"valid loss: {:.3f}".format(validation_loss),
						"total correct: {}".format(total_score),
						"accuracy: {:.3f}".format(total_score),
					)
            
					self.model.train()
					loss_per_x_batches = 0
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
	def abc(self):
		pass
    