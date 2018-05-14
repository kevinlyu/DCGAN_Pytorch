import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime


class Generator(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.ReLU())
        # 4 x 4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.ReLU())
        # 8 x 8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf),
                                    nn.ReLU())

        # 16 x 16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # 32 x 32
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0)) # for LSGAN

        '''
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),
                                    nn.Sigmoid()) # for vanilla GAN 
        '''

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='output/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)  # print arguments

# if output folder dos not exist, create it
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# random seed for pytorch
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# speed up GPU computation
cudnn.benchmark = True

# dataloader
data_transform = transforms.Compose([
    transforms.Resize(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize(mean, variance)
])

data_dir = "./dataset/animation/"
dset = dset.ImageFolder(data_dir, data_transform)
loader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, shuffle=True, num_workers=8)

# Model
ndf = opt.ndf  # number of filters in discriminator
ngf = opt.ngf  # number of filters in generator
nc = 3  # channels of input image

netD = Discriminator(nc, ndf)
netG = Generator(nc, ngf, opt.nz)

if (opt.cuda):
    netD.cuda()
    netG.cuda()

# loss and optimizer
criterion = nn.BCELoss()  # binary cross entropy
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=opt.lr)

# optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr)
# optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr)

# optimizerD = torch.optim.Adamax(netD.parameters(), lr=opt.lr)
# optimizerG = torch.optim.Adamax(netG.parameters(), lr=opt.lr)


# global variables
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

noise = Variable(noise)
real = Variable(real)
label = Variable(label)

# if GPU is available, move data to GPU memory
if (opt.cuda):
    noise = noise.cuda()
    real = real.cuda()
    label = label.cuda()

# use a list to record loss value
d_loss = []
g_loss = []

# time counter
start_time = datetime.datetime.now()

# training
for epoch in range(1, opt.niter + 1):
    for i, (images, _) in enumerate(loader):
        # fDx
        netD.zero_grad()
        # train with real data, resize real because last batch may has less than
        # opt.batchSize images
        real.data.resize_(images.size()).copy_(images)
        label.data.resize_(images.size(0)).fill_(real_label)

        output = netD(real)

        #errD_real = criterion(output, label) # vanilla GAN
        errD_real = 0.5 * torch.mean((output - label) ** 2)  # LSGAN
        errD_real.backward()

        # train with fake data
        label.data.fill_(fake_label)
        noise.data.resize_(images.size(0), opt.nz, 1, 1)
        noise.data.normal_(0, 1)

        fake = netG(noise)
        # detach gradients here so that gradients of G won't be updated
        output = netD(fake.detach())
        #errD_fake = criterion(output, label) # vanilla GAN
        errD_fake = 0.5 * (torch.mean((output - label)) ** 2)  # LSGAN
        errD_fake.backward()

        errD = errD_fake + errD_real
        optimizerD.step()

        # fGx
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake)
        #errG = criterion(output, label) # vanilla GAN
        errG = 0.5 * (torch.mean(output - label) ** 2)  # LSGAN
        errG.backward()
        optimizerG.step()

        # print log info
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f '
              % (epoch, opt.niter, i, len(loader),
                 errD.data[0], errG.data[0]))

        d_loss.append(errD.data[0])
        g_loss.append(errG.data[0])

        # visualize
        if (i % 100 == 0):
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d_iter%03d.png' % (opt.outf, epoch, i),
                              normalize=True)

end_time = datetime.datetime.now()

print("spent {} minutes".format((end_time - start_time).seconds / 60))
torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))

# plot learning curve
plt.plot(g_loss, label="Generator")
plt.plot(d_loss, label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig(opt.outf + "loss")
plt.close()

# save loss as .txt file
np.savetxt(opt.outf + "g_loss.txt", g_loss)
np.savetxt(opt.outf + "d_loss.txt", d_loss)

# Smoothing loss curve
N = 100  # moving window size
g_loss_smooth = np.convolve(g_loss, np.ones((N,)) / N, mode='valid')
d_loss_smooth = np.convolve(d_loss, np.ones((N,)) / N, mode='valid')

np.savetxt(opt.outf + "g_loss_smooth.txt", g_loss_smooth)
np.savetxt(opt.outf + "d_loss_smooth.txt", d_loss_smooth)

plt.plot(g_loss_smooth, label="Generator")
plt.plot(d_loss_smooth, label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig(opt.outf + "loss(smooth)")
plt.close()
