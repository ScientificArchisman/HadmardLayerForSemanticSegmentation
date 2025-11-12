import torch
from torch import nn 


def get_hadamard_matrix(k: int, *, dtype=torch.float32):
        """
        Sylvester construction of a 2^k x 2^k Hadamard matrix with values ±1.
        Runs on GPU if device='cuda' (or if CUDA is available and device is None).
        """
        if k < 0:
            raise ValueError("k must be >= 0")

        H = torch.ones((1, 1),  dtype=dtype)
        for _ in range(k):
            H = torch.cat([torch.cat([H,  H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
        return H

def make_codebook(n_rows, k: int, row_idx=None, dtype=torch.float32, ortho=True):
    H = get_hadamard_matrix(k=k, dtype=dtype)

    if row_idx is None:
        row_idx = torch.arange(n_rows)
    codebook = H[row_idx]

    if ortho:
        codebook = codebook / (2 ** (k/2))
    return codebook

class HadamardLayer(nn.Module):
    def __init__(self, k, n_rows):
        super().__init__()
        C = make_codebook(n_rows=n_rows, k=k, dtype=torch.float32, ortho=True)  
        self.register_buffer("C", C.contiguous(), persistent=True)          
        self.register_buffer("Ct", C.t().contiguous(), persistent=True)    

    def encode(self, y):     
        # (N,C,H,W) x (C,K) -> (N,K,H,W)
        return torch.einsum('nchw,ck->nkhw', y, self.C)

    def decode(self, z):     
        # (N,K,H,W) x (K,C) -> (N,C,H,W)
        return torch.einsum('nkhw,kc->nchw', z, self.Ct)

    def forward(self, y):
        z = self.encode(y)     # (N,K,H,W)
        yhat = self.decode(z)  # (N,C,H,W)
        return yhat
        


    

if __name__ == "__main__":
    y = torch.rand(1, 19, 128, 128)
    codebook = make_codebook(n_rows=y.shape[1], k = 5)
    model = HadamardLayer(k=5, n_rows=y.shape[1])
    yhat = model(y)
    print(f"Yhat shape: {yhat.shape}")



