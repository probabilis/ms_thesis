import torch
import matplotlib.pyplot as plt


def wall_mask_from_labels(L):
    """
    # checks neighbour change of 2D input tensor Ls
    # L: bool (H,W) / either for positive u or negative u
    # returns boolean matrix W

    same implementation functionality as but way faster: 

    H, W = L.shape
    W_mask = torch.zeros_like(L, dtype=torch.bool)

    # check each cell against its left neighbor
    for i in range(H):
        for j in range(1, W):
            if L[i, j] != L[i, j - 1]:
                W_mask[i, j] = True

    # check each cell against its top neighbor
    for i in range(1, H):
        for j in range(W):
            if L[i, j] != L[i - 1, j]:
                W_mask[i, j] = True
    """

    W = torch.zeros_like(L, dtype=torch.bool)
    W[:, 1:] |= (L[:, 1:] != L[:, :-1]) # 

    W[1:, :] |= (L[1:, :] != L[:-1, :])
    return W



@torch.no_grad()
def quality_scores(u_exp, u_opt, blur_sigma=1.0):
    """
    Fisher discriminant & Perimeter functional
    """
    # labels from optimized solution
    L = (u_opt > 0)

    """
    https://www.wikiwand.com/en/articles/Linear_discriminant_analysis
    """
    u_pos = u_exp[L] # mask for positive values
    u_neg = u_exp[~L] # mask for negative values


    mu_pos, mu_neg = u_pos.mean(), u_neg.mean()

    #torch.tensor.numel() returns the total number of elements of tensor
    var_pos = u_pos.var(unbiased=True) if u_pos.numel() > 1 else torch.tensor(0., device=u_exp.device)
    var_neg = u_neg.var(unbiased=True) if u_neg.numel() > 1 else torch.tensor(0., device=u_exp.device)

    try:
        fisher_J = (mu_pos - mu_neg).pow(2) / (var_pos + var_neg)   
    except ValueError:
        fisher_J = 0

    # boundary calculation
    W = wall_mask_from_labels(L)
    PLOT = False
    if PLOT:
        plt.figure()
        plt.imshow(W.float(), origin="lower")
        plt.show()
    
    perimeter = W.float().sum()

    return float(fisher_J), float(perimeter), W.float()





