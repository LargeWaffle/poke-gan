import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torchvision.utils as vutils


def plot_images(dataloader, device):
    # Plot some training images
    real_img_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_img_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


def plot_losses(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses, label="D")
    plt.plot(g_losses, label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_gif(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=2000, repeat_delay=1000, blit=True)

    ani.save('pokegan.gif', writer="pillow", fps=12)


def side_by_side(dataset, device, img_list):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataset))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
