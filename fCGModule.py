"""
Original code: https://github.com/zlin7/CGNet/blob/master/CGNet/cudaCG/cuda/fCG.py

Copyright (c) <2018> <Zhen Lin, Shubhendu Trivedi, Risi Kondor>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from torch import nn
import torch

import CG_cuda

torch.manual_seed(42)
import numpy as np


class fCGFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, taus, activations, output_length):
        taus = torch.tensor(taus, dtype=torch.int)
        ctx.save_for_backward(taus, activations)
        output = torch.zeros(activations.shape[0],
                             output_length,
                             2,
                             device=torch.device('cuda'),
                             dtype=torch.float,
                             requires_grad=True)
        CG_cuda.forward(activations.contiguous(), output, taus.shape[0] - 1, activations.shape[0], taus)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        taus, activations = ctx.saved_tensors
        grad_input = torch.zeros(activations.shape, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
        CGlength = 0
        maxL = taus.shape[0] - 1
        for l1 in range(maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1 - l2, min(l1 + l2, maxL) + 1):
                    CGlength += (2 * l + 1) * (2 * l2 + 1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        CG_cuda.backward(activations, grad_input, grad_output, CGspace, maxL, activations.shape[0], taus)
        del CGspace
        return None, grad_input, None


class fCGModule(nn.Module):
    def __init__(self, maxL, taus):
        super(fCGModule, self).__init__()
        self.maxL = maxL
        self.taus = taus
        self.cum_taus = np.concatenate([[0], (self.taus * (1+2*np.arange(self.maxL+1))).cumsum()])
        self.new_tau = self.cal_new_tau(taus)
        self.cum_new_tau = np.concatenate([[0], (self.new_tau * (1+2*np.arange(self.maxL+1))).cumsum()])

    def cal_new_tau(self, taus):
        new_tau = np.zeros(self.maxL + 1, dtype=int)
        for l1 in range(self.maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1-l2, min(self.maxL, l1+l2)+1):
                    new_tau[l] += taus[l1] * taus[l2]
        return new_tau

    def forward(self, activations):
        assert(activations.is_cuda)
        output = fCGFunction.apply(self.taus, activations.contiguous(), self.cum_new_tau[-1])
        return output
