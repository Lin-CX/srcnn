from dataset import My_Dataset

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import time
from PIL import Image


class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.patch_extraction = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.non_linear = nn.Conv2d(64, 32, kernel_size=1)
        self.reconstruction = nn.Conv2d(32, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = F.relu(self.patch_extraction(x))
        x = F.relu(self.non_linear(x))
        out = torch.sigmoid(self.reconstruction(x))
        return out


def tensor_to_PIL(tensor):

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image



# check availability of GPU
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

# some parameters
crop_size = 256
batch_size = 4
lr = 1.5e-2
PATH = './net_params.pkl'
is_training = False

# preprocessing
tsf = transforms.Compose([transforms.CenterCrop(crop_size),
                        transforms.GaussianBlur(15, 1),
                        transforms.ToTensor()])
target_tsf = transforms.Compose([transforms.CenterCrop(crop_size),
                        transforms.ToTensor()])

trainSet = My_Dataset(root='./BSDS300', transform=tsf, target_transform=target_tsf)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

testSet = My_Dataset(root='./BSDS300', train=False, transform=tsf, target_transform=target_tsf)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False)

# define network
net = SRCNN()

srcnn = net.to(dev)

# loss function and optimizer
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(srcnn.parameters(), lr=lr, momentum=0.9)

# start training
t_start = time.time()

if is_training:
    for epoch in range(1500):

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data[0].to(dev), data[1].to(dev)

            # zero the parameter gradients
            optimizer.zero_grad()

            # perform forward pass
            outputs = srcnn(inputs)

            # set loss
            loss = mse_loss(outputs, labels)

            # backprop
            loss.backward()

            # SGD step
            optimizer.step()

            # save loss
            running_loss += loss.item()

        if epoch % 20 == 19:
            print("[%d]\tloss:\t%f" % (epoch+1, running_loss), end="\t")
            t_end = time.time()
            print("elapsed: %f sec" % (t_end-t_start))
            #print('elapsed:', t_end-t_start, 'sec')
            t_start = t_end
            if epoch % 100 == 99:
                img_index = 0
                temp = tensor_to_PIL(inputs[img_index])
                temp.save('./input_train.jpg')
                temp = tensor_to_PIL(outputs[img_index])
                temp.save('./output_train.jpg')
                temp = tensor_to_PIL(labels[img_index])
                temp.save('./label_train.jpg')


    print("Finished Training")

    # save the parameters in network
    torch.save(net.state_dict(),PATH)
    print("Saved parameters")

else:
    # load the parameters in network
    srcnn=srcnn.load_state_dict(torch.load(PATH))

total_loss = 0.0

# testing
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)

        loss = mse_loss(outputs, labels)
        total_loss += loss.item()

        if True:
            img_index = 3
            temp = tensor_to_PIL(images[img_index])
            temp.save('./input.jpg')
            temp = tensor_to_PIL(outputs[img_index])
            temp.save('./output.jpg')
            temp = tensor_to_PIL(labels[img_index])
            temp.save('./label.jpg')
            break

print("Overall Loss: %f" % (total_loss))