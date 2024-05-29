import argparse
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import utils, datasets, transforms
#from torch.utils.tensorboard import SummaryWriter

# 初始化一个writer
#writer = SummaryWriter('runs/your_experiment_name')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoches", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--type", type=str, default='GAN', help="The type of GAN")
    parser.add_argument("--type", type=str, default='DCGAN', help="The type of DCGAN")
    return parser.parse_args()

class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape)
        return x


class VAE(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_shape, latent_dim)
        self.decoder = Decoder(latent_dim, img_shape)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def train_vae(vae, data_loader, epochs, device):
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    vae.to(device)

    for epoch in range(epochs):
        vae.train()
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon_imgs, mu, log_var = vae(imgs)
            recon_loss = F.mse_loss(recon_imgs, imgs)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator_CNN, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))  # 100 ——> 128 * 8 * 8 = 8192

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_CNN(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator_CNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),

        )

        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())  # 128 * 2 * 2 ——> 1

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def train():
    opt = args_parse()

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    data = datasets.ImageFolder('./data', transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True)

    img_shape = (opt.channels, opt.img_size, opt.img_size)
    if opt.type == 'DCGAN':
        generator = Generator_CNN(opt.latent_dim, img_shape)
        discriminator = Discriminator_CNN(img_shape)
    else:
        generator = Generator(opt.latent_dim, img_shape)
        discriminator = Discriminator(img_shape)

    adversarial_loss = torch.nn.BCELoss()

    cuda = torch.cuda.is_available()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    device = torch.device("cuda" if cuda else "cpu")
    # 初始化VAE并训练
    vae = VAE((opt.channels, opt.img_size, opt.img_size), opt.latent_dim)
    train_vae(vae, train_loader, 5, device)  # opt.n_epoches=50
    generator.load_state_dict(vae.decoder.state_dict(), strict=False)
    # GAN训练设置
    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)

    print(generator)
    print(discriminator)
    discriminator_step = 2
    # Training
    for epoch in range(opt.n_epoches):
        for i, (imgs, _) in enumerate(train_loader):
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            real_imgs = imgs.to(device)

            # Train Discriminator more frequently than Generator
            # Update the discriminator 5 times per generator update
            for _ in range(discriminator_step):
                 optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                 real_loss = adversarial_loss(discriminator(real_imgs), valid)
                # Generate a batch of images:q
                 z = torch.randn(imgs.size(0), 100, device=device)
                 gen_imgs = generator(z)
               # writer.add_images('Generated Images', gen_imgs, global_step=batches_done)
                 fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                 d_loss = (real_loss + fake_loss) / 2
                 d_loss.backward()
                 optimizer_D.step()
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            print(
            f"[Epoch {epoch}/{opt.n_epoches}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G Loss: {g_loss.item()}]")
            #writer.add_scalar('Loss/Generator', g_loss.item(), global_step=batches_done)
            #writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step=batches_done)
            os.makedirs("images_3", exist_ok=True)
            batches_done = epoch * len(train_loader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images_3/%d.png" % batches_done, nrow=5, normalize=True)

    print("Training complete")


if __name__ == '__main__':
    train()
