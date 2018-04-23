import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

class Model:
	def __init__(self, arch, units, lr):
		self.arch = arch
		self.units = units
		self.lr = lr
	def create(self):
		self.model = getattr(models, self.arch)(pretrained=True)

		for param in self.model.parameters():
			param.requires_grad = False

		self.model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, self.units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(self.units, 102)),
                          ('relu', nn.ReLU()),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

	def train(self, gpu, epochs, save_dir, train_loader, valid_loader, num_train_images, num_valid_images):
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
                ### -------------------------------------------------------------------------------- ###
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
                ### -------------------------------------------------------------------------------- ###
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
						"accuracy: {:.3f}".format(total_score/num_valid_images),
					)
            
					self.model.train()
					loss_per_x_batches = 0
