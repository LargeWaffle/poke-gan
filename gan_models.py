import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torchvision.utils as vutils

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

lr = 0.0002

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# Generator
# ~~~~~~~~~
#
# The generator, :math:`G`, is designed to map the latent space vector
# (:math:`z`) to data-space. Since our data are images, converting
# :math:`z` to data-space means ultimately creating a RGB image with the
# same size as the training images (i.e. 3x64x64). In practice, this is
# accomplished through a series of strided two dimensional convolutional
# transpose layers, each paired with a 2d batch norm layer and a relu
# activation. The output of the generator is fed through a tanh function
# to return it to the input data range of :math:`[-1,1]`. It is worth
# noting the existence of the batch norm functions after the
# conv-transpose layers, as this is a critical contribution of the DCGAN
# paper. These layers help with the flow of gradients during training. An
# image of the generator from the DCGAN paper is shown below.
#
# .. figure:: /_static/img/dcgan_generator.png
#    :alt: dcgan_generator
#
# Notice, how the inputs we set in the input section (*nz*, *ngf*, and
# *nc*) influence the generator architecture in code. *nz* is the length
# of the z input vector, *ngf* relates to the size of the feature maps
# that are propagated through the generator, and *nc* is the number of
# channels in the output image (set to 3 for RGB images). Below is the
# code for the generator.
#
######################################################################
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, model_input):
        return self.main(model_input)


class UpsampledGenerator(nn.Module):
    def __init__(self, ngpu):
        super(UpsampledGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True),
            nn.Conv2d(nz, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, model_input):
        return self.main(model_input)


######################################################################
# Discriminator
# ~~~~~~~~~~~~~
#
# As mentioned, the discriminator, :math:`D`, is a binary classification
# network that takes an image as input and outputs a scalar probability
# that the input image is real (as opposed to fake). Here, :math:`D` takes
# a 3x64x64 input image, processes it through a series of Conv2d,
# BatchNorm2d, and LeakyReLU layers, and outputs the final probability
# through a Sigmoid activation function. This architecture can be extended
# with more layers if necessary for the problem, but there is significance
# to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
# DCGAN paper mentions it is a good practice to use strided convolution
# rather than pooling to downsample because it lets the network learn its
# own pooling function. Also batch norm and leaky relu functions promote
# healthy gradient flow which is critical for the learning process of both
# :math:`G` and :math:`D`.
#
#########################################################################
# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, model_input):
        return self.main(model_input)


def multi_gpu(network, device, nb_gpu):
    n = network
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (nb_gpu > 1):
        n = nn.DataParallel(network, list(range(nb_gpu)))

    return n


def get_generator(device, nb_gpu, upsampled=True):
    # Create the generator
    if upsampled:
        generator_network = UpsampledGenerator(nb_gpu).to(device)
    else:
        generator_network = Generator(nb_gpu).to(device)

    generator_network = multi_gpu(generator_network, device, nb_gpu)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    generator_network.apply(weights_init)

    # Print the model
    print(generator_network)

    optimizer_g = optim.Adam(generator_network.parameters(), lr=lr, betas=(beta1, 0.999))

    return generator_network, optimizer_g


def get_discriminator(device, nb_gpu):
    # Create the Discriminator
    discriminator_network = Discriminator(nb_gpu).to(device)
    discriminator_network = multi_gpu(discriminator_network, device, nb_gpu)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator_network.apply(weights_init)

    # Print the model
    print(discriminator_network)

    optimizer_d = optim.Adam(discriminator_network.parameters(), lr=lr, betas=(beta1, 0.999))

    return discriminator_network, optimizer_d


def train(dataloader, discriminator_network, optimizer_d, generator_network, optimizer_g, device, *, num_epochs=100):
    # Training Loop

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Lists to keep track of progress
    imgs = []
    g_losses = []
    d_losses = []
    iters = 0

    len_ds = len(dataloader)
    status_step = len_ds // 2

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            discriminator_network.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator_network(real_cpu).view(-1)
            # Calculate loss on all-real batch
            err_d_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            err_d_real.backward()
            d_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator_network(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator_network(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            err_d_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err_d_fake.backward()
            d_g_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            err_d = err_d_real + err_d_fake
            # Update D
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_network.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator_network(fake).view(-1)
            # Calculate G's loss based on this output
            err_g = criterion(output, label)
            # Calculate gradients for G
            err_g.backward()
            d_g_z2 = output.mean().item()
            # Update G
            optimizer_g.step()

            # Output training stats
            if i % status_step == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))

            # Save Losses for plotting later
            g_losses.append(err_g.item())
            d_losses.append(err_d.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len_ds - 1)):
                with torch.no_grad():
                    fake = generator_network(fixed_noise).detach().cpu()
                imgs.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return imgs, d_losses, g_losses
