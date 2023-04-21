from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from infmodels import Generator32, Generator64, nz


def remove_mkeys(model, dic):
    new_state_dict = OrderedDict()

    for k, v in dic.items():
        new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict)

    return model


ngpu = 0
nb_pokemons = 16

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Running on", device)

netG = Generator64(ngpu).to(device)
dc = torch.load('models/64/generator.pt', map_location=device)

netG = remove_mkeys(netG, dc)
netG.eval()

with torch.no_grad():
    noise = torch.randn(nb_pokemons, nz, 1, 1, device=device)

    fake = netG(noise).detach().cpu()

img_list = [vutils.make_grid(fake, padding=2, normalize=True)]

plt.figure(figsize=(12, 6))
plt.axis("off")
plt.title("Fake Pokemons")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
