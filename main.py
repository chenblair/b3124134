# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import inception
from data.mnist import MNIST
import argparse, sys
import numpy as np
import datetime
import shutil
from tqdm import tqdm



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.accuracy=[]
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 100, 5, 1)
        # self.conv4 = nn.Conv2d(100, 200, 5, 1)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 8, 8)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 8, 8)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 4, 4)
        # x = F.relu(self.conv4(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 400)
        #x = x.view(x.size(0), x.size(1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x#F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch, eps=9.9,nums=10):
    model.train()
   # for batch_idx, (data, target, idx, is_pure, is_corrupt) in enumerate(train_loader):
    loss_a=[]
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.softmax(output, dim=1)
        # print(output[:,nums])
        output = (output + (output[:,nums].unsqueeze(1) / eps) + 1E-10).log()
        # output.log_() #= F.log_softmax(output, dim=1)
        # print(output, target)
        loss = F.nll_loss(output, target)
        # print(output, target, loss)
        loss_a.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(loss_a), torch.mean(output[:,nums])

def test(args, model, device, test_loader,nums):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output[:,:nums].argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc=100. * correct / len(test_loader.dataset);
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc
    
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
    parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
    parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--top_bn', action='store_true')
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'mnist')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--num_iter_per_epoch', type=int, default=400)
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--eps', type=float, default=9.9)
    parser.add_argument('--load_model', type=str, default="")
    
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    batch_size=args.batch_size
    
    if args.dataset=='mnist':
        input_channel=1
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200
        train_dataset = MNIST(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                             )

        test_dataset = MNIST(root='./data/',
                                   download=True,  
                                   train=False, 
                                   transform=transforms.ToTensor(),
                                   noise_type=args.noise_type,
                                   noise_rate=args.noise_rate
                            )
    
    if args.dataset=='cifar10':
        input_channel=3
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200
        train_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                               )

        test_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                              )
    if args.dataset=='openimages':
        input_channel=3
        num_classes=16
        args.top_bn = False
        args.epoch_decay_start = 80
        openimages_transform = transforms.Compose([transforms.Resize((299, 299)),
                                transforms.ToTensor(),
                                transforms.Normalize([0, 0, 0], [1, 1, 1])])
        train_dataset = datasets.ImageFolder("/home/paul/noisy_labels/OIDv4_ToolKit/OID/Dataset_nl/train", transform=openimages_transform)
        test_dataset = datasets.ImageFolder("/home/paul/noisy_labels/OIDv4_ToolKit/OID/Dataset_nl/test", transform=openimages_transform)
        print("Train Dataset: {}\n Test Dataset: {}".format(len(train_dataset), len(test_dataset)))
        # test_dataset, _ = torch.utils.data.random_split(test_dataset, [100, len(test_dataset)-100])
        # test_dataset = datasets.ImageFolder("/Users/cherry/Documents/OIDv4_ToolKit/OID/Dataset_nl/train", transform=openimages_transform)


    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

    # noise_or_not = train_dataset.noise_or_not
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    #print('building model...')
    #cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes+1)
    #cnn1.cuda()
    #print(cnn1.parameters)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}".format(device))
    # cnn1 = Net().to(device)
    cnn1 = inception.inception_v3(pretrained=False, num_classes=num_classes + 1, aux_logits=False).to(device)
    #cnn1=nn.DataParallel(cnn1,device_ids=[0,1,2,3]).cuda()
        #print(model.parameters)
    #optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(cnn1.parameters(), lr=args.lr)
    
    optimizer = torch.optim.Adam(cnn1.parameters(), lr=args.lr)
    #optimizer = nn.DataParallel(optimizer, device_ids=[0,1,2,3]) 

    acc=[]
    loss=[]
    loss_pure=[]
    loss_corrupt=[]
    out=[]
    eee=1-args.noise_rate
    criteria =(-1)* (eee * np.log(eee) + (1-eee) * np.log((1-eee)/(args.eps-1)))
    name=str(args.dataset)+"_"+str(args.noise_type)+"_"+str(args.noise_rate)+"_"+str(args.eps)+"_"+str(args.seed)

    if (args.load_model != ""):
        checkpoint = torch.load(args.load_model)
        cnn1.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        acc = list(checkpoint['test_acc'])
        loss = list(checkpoint['loss'])

    for epoch in range(1, args.n_epoch + 1):
        l1,out10=train(args, cnn1, device, train_loader, optimizer, epoch, eps=args.eps, nums=num_classes)
        loss.append(l1)
        out.append(out10)
        acc.append(test(args, cnn1, device, test_loader,num_classes))
        print(l1,criteria)
        # if l1<criteria:
        #     break;
        torch.save({
            'model_state_dict': cnn1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'test_acc': acc
            }, "early_stopping/"+name+"_model.npy")
    save_dir="early_stoppig"
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)
    
    print(name)
    np.save("early_stopping/"+name+"_acc.npy",acc)
    np.save("early_stopping/"+name+"_loss.npy",loss)

if __name__=='__main__':
    main()
