from pattern_formation import fourier_multiplier
import torch

import matplotlib.pyplot as plt

from env_utils import plotting_style



def plot_fourier_multiplier():

    plotting_style()

    N = 10
    steps = 1000

    x = torch.arange(-N,+N,1/steps)
    print(x)
    y = fourier_multiplier(x)

    plt.plot(x,y, color = "black", linewidth = 3)
    plt.grid(color = "gray")
    plt.xlabel("$x$")
    plt.ylabel("$\\sigma(x)$")
    plt.show()
    
def gradient():

    N = 128

    u = torch.rand((N,N))

    ones = torch.ones_like(u)

    u = u + torch.sin(ones)

    fig, axs = plt.subplots(1,4)
    
    ux = uy = u

    plt.ion()

    for ii in range(100):
        print("ii", ii)
        ux = u - torch.roll(u, 1, 0)
        uy = u - torch.roll(u, 1, 1)

        u_grad = torch.sqrt(ux*ux + uy*uy)

        axs[0].imshow(u)
        axs[1].imshow(ux)
        axs[2].imshow(uy)
        axs[3].imshow(u_grad)

        u = u_grad

        plt.pause(1)

    plt.ioff()
    #plt.show()
    

if __name__ == "__main__":

    #plot_fourier_multiplier()  
    gradient()