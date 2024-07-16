""" Standalone version of Structured (Sequence) State Space (S4) model. """


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe

from src.models.nn import DropoutNd

import numpy as np

import precision_tools as pt

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=0.001, **kernel_args):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        if 'A_quant' in kernel_args and kernel_args['A_quant'] is not None:
            self.A_quant = int(kernel_args['A_quant'])
        else:
            self.A_quant = None

        if 'C_quant' in kernel_args and kernel_args['C_quant'] is not None:
            self.C_quant = int(kernel_args['C_quant'])
        else:
            self.C_quant = None

        if 'dt_quant' in kernel_args and kernel_args['dt_quant'] is not None:
            self.dt_quant = int(kernel_args['dt_quant'])
        else:
            self.dt_quant = None

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        if self.dt_quant is not None:
            dt = torch.exp(self.log_dt - (self.log_dt - max_quant_fn(self.log_dt, self.dt_quant)).detach()) # (H)
        else:
            dt = torch.exp(self.log_dt)

        if self.C_quant is not None:
            C = _r2c(self.C - (self.C - max_quant_fn(self.C, quant_levels=self.C_quant)).detach())
        else:
            C = _r2c(self.C) # (H N)
        
        if self.A_quant is not None:
            A_real_quant = -torch.exp(self.log_A_real) - (-torch.exp(self.log_A_real) - max_quant_fn(-torch.exp(self.log_A_real), quant_levels=self.A_quant)).detach()
            A_imag_quant = self.A_imag - (self.A_imag - max_quant_fn(self.A_imag, quant_levels=self.A_quant)).detach()
            A = A_real_quant + 1j * A_imag_quant # (H N)
        else:
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)

        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K
    
    def analysis(self):
        return ((-torch.exp(self.log_A_real), max_quant_fn(-torch.exp(self.log_A_real), quant_levels=self.A_quant)), # A_real
                (self.A_imag, max_quant_fn(self.A_imag, quant_levels=self.A_quant)),    # A_imag
                (self.C, max_quant_fn(self.C, quant_levels=self.C_quant)),              # C
                (torch.exp(self.log_dt), torch.exp(max_quant_fn(self.log_dt, self.dt_quant))))  # dt

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        if 'kernel_quant' in kernel_args and kernel_args['kernel_quant'] is not None:
            self.kernel_quant=int(kernel_args['kernel_quant'])
        else:
            self.kernel_quant = None
        if 'linear_quant' in kernel_args and kernel_args['linear_quant'] is not None:
            self.linear_quant=int(kernel_args['linear_quant'])
        else:
            self.linear_quant = None
        if 'act_quant' in kernel_args and kernel_args['act_quant'] is not None:
            self.act_quant=int(kernel_args['act_quant'])
        else:
            self.act_quant = None
        if 'state_quant' in kernel_args and kernel_args['state_quant'] is not None:
            self.state_quant=int(kernel_args['state_quant'])
        else:
            self.state_quant = None
            
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if self.linear_quant is not None:
            self.output_linear = nn.Sequential(
                pt.QuantizedConv1d(self.h, 2*self.h, kernel_size=1, quant_fn=pt.max_quant_fn, quant_levels=self.linear_quant),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2*self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        if self.state_quant is not None:
            k = k - (k - max_quant_fn(k, quant_levels=int(self.state_quant / 2))).detach()
            u = u - (u - max_quant_fn(u, quant_levels=int(self.state_quant / 2))).detach()

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        #np.savetxt("./checkpoint/kernel_f.txt", k_f.cpu().detach().numpy())
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if self.act_quant is not None:
            y = y - (y - max_quant_fn(y, quant_levels=self.act_quant)).detach()
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified


# taken from bitnet 1.58b
def max_quant_fn(a, quant_levels=2):
    if quant_levels is None:
        return a
    # scaling parameter to get an estimate of the magnitude of the activations. 
    # clamp to avoid division by zero
    #import pdb
    #pdb.set_trace()
    scale = quant_levels / 2 / torch.clamp(torch.max(a.abs().flatten(), dim=-1, keepdim=True)[0], min=1e-5) 

    # a * scale normalizes a. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    a_out = torch.clamp((a * scale).round(), min=-quant_levels // 2, max=quant_levels // 2) / scale
    return a_out

# taken from bitnet 1.58b
def mean_quant_fn(w, quant_levels=2):
    # scaling parameter to get an estimate of the magnitude of the weights. 
    # clamp to avoid division by zero
    scale = quant_levels / 2 / w.abs().flatten().mean().clamp(min=1e-5) 

    # w * scale normalizes w. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    w_out = (w * scale).round().clamp(-quant_levels // 2, quant_levels // 2) / scale
    return w_out