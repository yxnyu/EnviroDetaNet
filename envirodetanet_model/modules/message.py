from e3nn import o3
import torch
from torch import nn
from .edge_attention import Edge_Attention

class Message(nn.Module):
    '''Message Module'''
    def __init__(self,head,num_radial,num_features,irreps_sh,act):
        super(Message,self).__init__()
        self.feature=num_features
        self.Attention=Edge_Attention(head=head,num_radial=num_radial,num_features=num_features,act=act)
        irreps_mout = []
        instructions = []
        # Tensor product of irrep tensor features generated from scalar features and irrep tensor
        # Taking a scalar of 128 dimensions and a tensor of l=1,2,3,
        # each feature in the scalar is multiplied by the irrep tensor of '1o+2e+3o'.
        # The final result is '128x1o+128x2e+128x3o'
        # It is worth mentioning that each tensor product here is given a learnable weight.
        for i, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in o3.Irrep('0e') * ir_sh:
                k = len(irreps_mout)
                irreps_mout.append((num_features, ir_out))
                instructions.append((0, i, k, "uvu", True))
        self.tp = o3.TensorProduct(irreps_in1=o3.Irreps([(num_features, (0, 1))]), irreps_in2=irreps_sh,
                                   irreps_out=irreps_mout, instructions=instructions, shared_weights=True,
                                   internal_weights=True)

    def forward(self,S,rbf,sh,index):
        # First invariant features and radial features generate edge features(eij) by attention mechanism
        eij=self.Attention(S=S,rbf=rbf,index=index)
        # Then edge feature eij is divided into 2 parts,
        # one as the output of the invariant feature
        # and another tensor product with the irrep tensor generated by the spherical harmonic function
        # to output equivariant irreps feature
        mijs2,mijs=torch.split(eij,split_size_or_sections=[self.feature, self.feature],dim=-1)
        mijt=self.tp(mijs2,sh)
        return mijt,mijs
