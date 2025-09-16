import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import torchvision
import os

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

DATA_DIR = 'for_gan_mat'
SAVE_DIR = 'gansave'
os.makedirs(SAVE_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

# TensorBoard
writer = SummaryWriter('./TensorBoardX')

# -----------------------
# Argument Parser
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",    type=int,   default=1650)
parser.add_argument("--lr",          type=float, default=0.00001)
parser.add_argument("--b1",          type=float, default=0.0)
parser.add_argument("--b2",          type=float, default=0.9)
parser.add_argument("--n_cpu",       type=int,   default=8)
parser.add_argument("--latent_dim",  type=int,   default=1600)
parser.add_argument("--eeg_chans",   type=int,   default=19)
parser.add_argument("--csp_chans",   type=int,   default=19)
parser.add_argument("--time_pts",    type=int,   default=1280)
parser.add_argument("--n_critic",    type=int,   default=1)
parser.add_argument("--clip_value",  type=float, default=0.01)
parser.add_argument("--sample_interval", type=int, default=200)
opt = parser.parse_args()

# Dynamic sizes
EEG_CHANS   = opt.eeg_chans
CSP_CHANS   = opt.csp_chans
TIME_PTS    = opt.time_pts
LATENT_DIM  = opt.latent_dim

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
lambda_gp = 10

# -----------------------
# Model init
# -----------------------
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -----------------------
# Generator
# -----------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=1600):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.init_layer = nn.Sequential(
            nn.Linear(latent_dim, 256 * 1 * 20),
            nn.BatchNorm1d(256 * 1 * 20),
            nn.ReLU(True)
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # -> [128, 1, 40]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # -> [64, 1, 80]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # -> [32, 1, 160]
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=(1, 5), stride=(1, 4), padding=(0, 1)),  # -> [16, 1, 640]
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, kernel_size=(19, 5), stride=(1, 2), padding=(0, 0)),  # -> [1, 6, 1280]
            nn.Tanh()
        )

    def forward(self, z):
        out = self.init_layer(z)
        out = out.view(-1, 256, 1, 20)
        out = self.upconv(out)
        out = out[:, :, :, :1280]
        return out

# -----------------------
# Discriminator
# -----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # shared first conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,kernel_size=(1,23),padding=(0,11)),
            nn.LeakyReLU(0.2)
        )
        # EEG branch
        self.eeg_branch = nn.Sequential(
            nn.Conv2d(10,30,kernel_size=(EEG_CHANS,1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(30*(TIME_PTS),1)
        )
        # CSP branch
        self.csp_branch = nn.Sequential(
            nn.Conv2d(10,30,kernel_size=(CSP_CHANS,1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(30*(TIME_PTS),1)
        )

    def forward(self, x):
        f = self.conv1(x)
        if x.shape[2] == EEG_CHANS:
            return self.eeg_branch(f)
        elif x.shape[2] == CSP_CHANS:
            return self.csp_branch(f)
        else:
            raise ValueError("Unexpected channel size")
        print("Generator output shape:", x.shape)


def wgan(datatrain, cspdatatrain, label, nclass, nseed, sub_index, Cov, Dis_mean, Dis_std, P, B, Wb):

    # Initialize generator and discriminator
    discriminator = Discriminator()
    # discriminator2 = Discriminator()
    generator = Generator()
    discriminator.apply(weights_init)
    # discriminator2.apply(weights_init)
    generator.apply(weights_init)

    discriminator = discriminator.cuda()
    # discriminator2 = discriminator2.cuda()
    generator = generator.cuda()

    discriminator = discriminator.to(device)

    # discriminator2 = nn.DataParallel(discriminator2, device_ids=[0, 1, 2, 3, 4])
    generator = generator.to(device)
    discriminator.to(device)
    # discriminator2.to(device)
    generator.to(device)
    print('Generator')
    print(generator)
    print('Discriminator')
    print(discriminator)

    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    cspdatatrain = torch.from_numpy(cspdatatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, cspdatatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr*0.95, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr*5, betas=(opt.b1, opt.b2))
    # optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    Wb = torch.Tensor(Wb.transpose()).cuda()

    def compute_gradient_penalty(D, real_samples, fake_samples):

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))  # (5,1,1)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print (interpolates.shape)
        # interpolates = interpolates.cpu().detach().numpy()
        # interpolates = np.expand_dims(interpolates, axis=1)
        # interpolates = torch.from_numpy(interpolates).cuda()
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  Training
    # ----------
    new_data = []
    batches_done = 0
    discriminator.train()
    # discriminator2.train()
    generator.train()
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):  # 50 data

            imgs, csp_imgs, _ = data
            imgs = imgs.cuda()
            csp_imgs = csp_imgs.cuda()

            # Configure input
            real_data = Variable(imgs.type(Tensor))
            real_csp_data = Variable(csp_imgs.type(Tensor))

            if i % 1 == 0:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # optimizer_D2.zero_grad()
                # Sample noise as generator input
                # no noise, but use part of the origin input randomly

                # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                z = torch.randn(imgs.shape[0], 1600).cuda()
                # Generate a batch of images
                # !!! directly generate from randn
                fake_imgs = generator(z)

                # fake_csp_imgs = [Wb.mm(fake_imgs[fci_index, 0, :, :]) for fci_index in range(5)]
                fake_csp_imgs = torch.randn(fake_imgs.shape[0], 1, 19, 1280).cuda()
                for fci_index in range(fake_imgs.shape[0]):

                    fake_csp_imgs[fci_index, 0, :, :] = Wb.mm(fake_imgs[fci_index, 0, :, :1280])
                # Real images
                # ttt = discriminator(fake_csp_imgs)
                real_validity = discriminator(real_data)
                real_csp_validity = discriminator(real_csp_data)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                fake_csp_validity = discriminator(fake_csp_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_imgs.data)
                csp_gradient_penalty = compute_gradient_penalty(discriminator, real_csp_data.data, fake_csp_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                writer.add_scalar('Train/Discriminator_eeg', d_loss, epoch)
                writer.flush()

                d_csp_loss = -torch.mean(real_csp_validity) + torch.mean(fake_csp_validity) + lambda_gp * csp_gradient_penalty

                writer.add_scalar('Train/Discriminator_csp', d_csp_loss, epoch)
                writer.flush()

                d_loss += d_csp_loss * 0.1
                d_loss.backward()
                optimizer_D.step()
                # d_csp_loss.backward()
                # optimizer_D2.step()
                # use for tensorboardX
                # dd = d_loss + d_csp_loss
                writer.add_scalar('Train/Discriminator', d_loss, epoch)
                writer.flush()
                torch.cuda.empty_cache()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                # -----------------
                #  Train Generator
                # -----------------

                # z = torch.randn(imgs.shape[0], 100).cuda()

                # Generate a batch of images
                fake_imgs = generator(z)

                if epoch > 1398:
                    print(epoch)
                    fake_data = fake_imgs.data[:25].cpu().numpy()
                    new_data.append(fake_data)


                fake_csp_imgs = torch.randn(fake_imgs.shape[0], 1, 19, 1280).cuda()
                for fci_index in range(fake_imgs.shape[0]):
                    fake_csp_imgs[fci_index, 0, :, :] = Wb.mm(fake_imgs[fci_index, 0, :, :])
                # writer.add_graph(generator, z)

                # the constrains of the covariance matrix and eigenvalue
                tmp_fake_imgs = np.array(fake_imgs.cpu().detach())
                cov_loss = []
                ev_loss = []
                for cov_index in range(imgs.shape[0]):
                    one_fake_imgs = tmp_fake_imgs[cov_index, 0, :, :]
                    oneone = np.dot(one_fake_imgs, one_fake_imgs.transpose())


                    one_cov = oneone/np.trace(oneone)
                    one_dis = np.sqrt(np.sum(np.power(one_cov - Cov, 2)))
                    one_cov_loss_vec = np.abs(one_dis - Dis_mean) / Dis_std

                    if np.all(one_cov_loss_vec <= 1):
                        one_cov_loss_vec = np.zeros_like(one_cov_loss_vec)


                    one_cov_loss = np.mean(one_cov_loss_vec)
                    cov_loss.append(one_cov_loss)

                    BTP = np.dot(B.transpose(), P)
                    one_ev = np.dot(BTP, one_cov)
                    one_ev = np.dot(one_ev, BTP.transpose())
                    one_ev_four = np.diag(one_ev)[0:4]
                    one_ev_loss = np.mean(one_ev_four)
                    one_ev_loss = np.abs(np.log(one_ev_loss))
                    ev_loss.append(one_ev_loss)

                cov_loss = np.mean(cov_loss).astype(np.float32)
                cov_loss = np.clip(cov_loss, 0, 10).astype(np.float32)
                writer.add_scalar('Train/G_Cov_loss', cov_loss, epoch)
                writer.flush()

                ev_loss = np.mean(ev_loss).astype(np.float32)
                ev_loss = np.clip(ev_loss, 0, 10).astype(np.float32)
                writer.add_scalar('Train/G_Ev_loss', ev_loss, epoch)
                writer.flush()

                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                fake_csp_validity = discriminator(fake_csp_imgs)
                g_loss = -torch.mean(fake_validity) - torch.mean(fake_csp_validity) * 0.1
                writer.add_scalar('Train/G_g_loss', g_loss, epoch)
                writer.flush()

                g_loss.data = g_loss.data + torch.tensor(3 * cov_loss).cuda() + torch.tensor(10 * ev_loss).cuda()
                g_loss.backward()
                optimizer_G.step()

                # use for tensorboardX
                writer.add_scalar('Train/Generator', g_loss, epoch)
                writer.flush()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # save the generate data each 5-epoch
                if epoch % 5 == 0:
                    save_fake_img = tmp_fake_imgs[0, 0, :, :]
                    # cv.imwrite('a path' + str(nclass) +
                    #            '_epoch_' + str(epoch) + '.png', save_fake_img)
                    plt.imshow(save_fake_img, cmap='Greys', aspect='auto', origin='lower')
                    plt.savefig(os.path.join(SAVE_DIR, f'plt{sub_index}_class{nclass}_epoch{epoch}.jpg'))


                # writer.add_graph(generator, z)
                grid0 = torchvision.utils.make_grid(fake_imgs[0, 0, :, :])
                writer.add_image('output fake data0', grid0, global_step=0)


    # writer.close()
    torch.save(discriminator, os.path.join(SAVE_DIR, f'S{sub_index}_D_class{nclass}.pth'))
    torch.save(generator,     os.path.join(SAVE_DIR, f'S{sub_index}_G_class{nclass}.pth'))
    discriminator.eval()
    generator.eval()


    new_data = np.concatenate(new_data)
    new_data = np.asarray(new_data)
    writer.close()
    return new_data

