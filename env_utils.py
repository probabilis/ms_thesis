import torch

def tensor_type(x):
    print("Is float : ",isinstance(x,torch.FloatTensor) )
    print("Is double : ", isinstance(x,torch.DoubleTensor) )