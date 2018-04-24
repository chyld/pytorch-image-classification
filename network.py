import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np


class Network:

    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###

    def __init__(self, arch, units, lr):
        # currently works with resnet, densenet & inception models
        self.arch, self.units, self.lr = arch, units, lr
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

        self.model.train()

        for epoch in range(epochs):
            for ii, (inputs, labels) in enumerate(train_loader):
                ### -------------------------------------------------------------------------------- ###
                inputs, labels = Variable(
                    inputs, requires_grad=True), Variable(labels)

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
                    validation_loss, total_correct, accuracy = 0, 0, 0

                    for _, (inputs, labels) in enumerate(valid_loader):
                        inputs, labels = Variable(
                            inputs, requires_grad=False, volatile=True), Variable(labels)

                        if cuda and gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        outputs = self.model.forward(inputs)
                        loss = self.criterion(outputs, labels)

                        validation_loss += loss.data[0]
                        probabilities = torch.exp(outputs).data
                        equality = (labels.data == probabilities.max(1)[1])
                        total_correct += equality.sum()
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                    print(
                        "epoch: {} batch: {}".format(epoch, ii),
                        "train loss: {:.3f}".format(loss_per_x_batches),
                        "valid loss: {:.3f}".format(validation_loss),
                        "total correct: {}".format(total_correct),
                        "accuracy: {:.3f}".format(accuracy),
                    )

                    self.model.train()
                    loss_per_x_batches = 0

    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###

    def save(self, save_dir, train_dataset):
        state = {'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'class_to_idx': train_dataset.class_to_idx,
                 'arch': self.arch,
                 'units': self.units,
                 'lr': self.lr
                 }
        torch.save(state, save_dir + '/checkpoint.pth')

    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###

    def predict(self, image, gpu, top_k, names):
        cuda = torch.cuda.is_available()

        # move the model to cuda
        if cuda and gpu:
            self.model.cuda()
        else:
            self.model.cpu()

        # turn dropout OFF
        self.model.eval()

        # convert numpy array to tensor
        image = torch.from_numpy(np.array([image])).float()

        # create variable from tensor
        image = Variable(image, requires_grad=False, volatile=True)

        # move the image to cuda
        if cuda and gpu:
            image = image.cuda()

        # forward propagation
        output = self.model.forward(image)

        # get probabilities
        probabilities = torch.exp(output).data

        # getting the topk probabilites and indexes
        top_p = torch.topk(probabilities, top_k)[0].tolist()[0]
        top_i = torch.topk(probabilities, top_k)[1].tolist()[0]

        # creating a reverse mapping from index to class
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}

        # converting the list of indexes to list of classes
        top_c = list(map(lambda i: idx_to_class[i], top_i))

        # convert list of classes to list of flower names
        top_n = [names[c] for c in top_c]

        return list(zip(top_p, top_n))

    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
    ### ------------------------------------------------------ ###
