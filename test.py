import torch


#U = torch.linspace(0,10,11)
U = torch.ones(10,10)
print(U)
print(U.shape)
FT = torch.fft.fft2(U)
print(FT)