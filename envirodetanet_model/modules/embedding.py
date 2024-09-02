import torch
from torch import nn, device, tensor, float32
from rdkit import Chem
from .acts import activations
from rdkit.Chem import AllChem
import json
import math
import numpy as np

def get_elec_feature(max_atomic_number,device):
    '''
    Electronic feature of the first 4 periodic elements.
     The number of single and paired electrons in an orbital.
     Also indicates the filled and half-filled state of the orbitals.
      '''
    elec_feature = tensor(
        # P:Pair electron
        # S:Single electron
        #|  1s |  2s |  2p |  3s |  3p |  4s |  3d |  4p |
        # P  S  P  S  P  S  P  S  P  S  P  S  P  S  P  S
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 0 None
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 1 H
         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 2 He
         [2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 3 Li
         [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 4 Be
         [2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 5 B
         [2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 6 C
         [2, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 7 N
         [2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 8 O
         [2, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 9 F
         [2, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 10 Ne
         [2, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 11 Na
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 12 Me
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],  # 13 Al
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, ],  # 14 Si
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, ],  # 15 P
         [2, 0, 2, 0, 6, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, ],  # 16 S
         [2, 0, 2, 0, 6, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, ],  # 17 Cl
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, ],  # 18 Ar
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, ],  # 19 K
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, ],  # 20 Ca
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, ],  # 21 Sc
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, ],  # 22 Ti
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, ],  # 23 V
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 5, 0, 0, ],  # 24 Cr
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 5, 0, 0, ],  # 25 Mn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 2, 4, 0, 0, ],  # 26 Fe
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 4, 3, 0, 0, ],  # 27 Co
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 6, 2, 0, 0, ],  # 28 Ni
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1,10, 0, 0, 0, ],  # 29 Cu
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 0, ],  # 30 Zn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 1, ],  # 31 Ga
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 2, ],  # 32 Ge
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 3, ],  # 33 As
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 2, 2, ],  # 34 Se
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 4, 1, ],  # 35 Br
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 6, 0, ],  # 36 Kr
         ], dtype=float32)
    elec=elec_feature[0:max_atomic_number+1]
    return (elec/elec.max()).to(device)

def get_atom_features(atom):
    features = []
    
    # Aromaticity (1 bit)
    features.append(int(atom.GetIsAromatic()))
    
    # Atom in ring (1 bit)
    features.append(int(atom.IsInRing()))
    
    # Atomic number (1 hot encoding, 7 bits)
    atomic_num = atom.GetAtomicNum()
    atomic_num_features = [0] * 7
    atomic_num_features[min(atomic_num, 6)] = 1
    features.extend(atomic_num_features)
    
    # Formal charge (7 bits)
    charge = atom.GetFormalCharge()
    charge_features = [0] * 7
    charge_features[charge + 3] = 1
    features.extend(charge_features)
    
    # Number of bonds (6 bits)
    num_bonds = len(atom.GetBonds())
    bond_features = [0] * 6
    bond_features[min(num_bonds - 1, 5)] = 1
    features.extend(bond_features)
    
    # Radical electrons (1 bit)
    features.append(int(atom.GetNumRadicalElectrons() > 0))
    
    # Hybridization (5 bits)
    hybridization = atom.GetHybridization()
    hybrid_features = [0] * 5
    hybrid_types = [Chem.rdchem.HybridizationType.SP, 
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2]
    if hybridization in hybrid_types:
        hybrid_features[hybrid_types.index(hybridization)] = 1
    features.extend(hybrid_features)
    
    # Chirality (3 bits)
    chirality = atom.GetChiralTag()
    chiral_features = [0] * 3
    if chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_features[0] = 1
    elif chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_features[1] = 1
    else:
        chiral_features[2] = 1
    features.extend(chiral_features)
    
    return np.array(features)

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)
        return output

class ClsAttention(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.attention = Attention(d_model).to(device)
        self.device = device
        
    def forward(self, cls_feat):
        cls_feat = cls_feat.to(self.device)
        return self.attention(cls_feat.unsqueeze(1)).squeeze(1)

class DynamicWeightAttention(nn.Module):
    def __init__(self, fingerprint_dim, cls_dim, output_dim):
        super(DynamicWeightAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(fingerprint_dim + cls_dim, (fingerprint_dim + cls_dim) // 2),
            nn.Tanh(),
            nn.Linear((fingerprint_dim + cls_dim) // 2, 2)
        )
        self.fingerprint_proj = nn.Linear(fingerprint_dim, output_dim)
        self.cls_proj = nn.Linear(cls_dim, output_dim)
    
    def forward(self, fingerprint, cls_vector):
        combined = torch.cat([fingerprint, cls_vector], dim=-1)
        attention_weights = torch.softmax(self.attention(combined), dim=-1)
        
        fp_proj = self.fingerprint_proj(fingerprint)
        cls_proj = self.cls_proj(cls_vector)
        
        weighted_fp = fp_proj * attention_weights[:, 0].unsqueeze(-1)
        weighted_cls = cls_proj * attention_weights[:, 1].unsqueeze(-1)
        
        return weighted_fp + weighted_cls, attention_weights

class Embedding(nn.Module):
    def __init__(self, num_features, act, max_atomic_number=9, device=device('cuda'), smiles=None):
        with open('/scratch/yx2892/qm9s_processed_data.json') as f1:
            self.qm9s_data = json.load(f1)
        with open('/scratch/yx2892/select_data.json') as f2:
            self.select_data = json.load(f2)
        self.json_data = {**self.qm9s_data, **self.select_data}
        super(Embedding, self).__init__()
        self.elec = get_elec_feature(max_atomic_number=max_atomic_number, device=device)
        self.act = activations(type=act, num_features=num_features)
        self.elec_emb = nn.Linear(16, num_features, bias=False)
        self.nuclare_emb = nn.Embedding(max_atomic_number+1, num_features)
        self.atom_supp_emb = nn.Linear(31, num_features, bias=False)
        self.ls = nn.Linear(num_features, num_features)
        self.device = torch.device(device)
        self.cls_attention = ClsAttention(d_model=512, device=device) 
        self.dynamic_attention = DynamicWeightAttention(fingerprint_dim=2048, cls_dim=512, output_dim=512)
        self.reset_parameters()
        self.smiles = smiles

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)
        self.ls.bias.data.fill_(0)
        self.nuclare_emb.reset_parameters()
        nn.init.xavier_uniform_(self.elec_emb.weight)
        nn.init.xavier_uniform_(self.atom_supp_emb.weight)

    def get_molecular_fingerprints(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    def forward(self, z, smiles=None):
        if smiles is None:
            # 如果 smiles 为 None，使用简化的计算
            return self.act(self.ls(self.nuclare_emb(z) + self.elec_emb(self.elec[z, :])))

        total_feat = torch.zeros_like(self.nuclare_emb(z)).to(self.device)
        start_index = 0
        
        # 确保 smiles 是一个列表
        if isinstance(smiles, str):
            smiles = [smiles]
        
        cleaned_smiles = [s for s in smiles if s is not None]
        attention_weights_list = []
        for smile in cleaned_smiles:
            if not isinstance(smile, str):
                smile = str(smile)
            print(f"SMILES 类型: {type(smile)}")
            mol = Chem.MolFromSmiles(smile)
            
            if mol is None:
                print(f"警告：无法从 SMILES 创建分子对象: {smile}")
                continue  # 跳过无法处理的 SMILES
            mol_with_h = Chem.AddHs(mol)
            smiles_with_h_count = mol_with_h.GetNumAtoms()
            end_index = start_index + smiles_with_h_count
            
            nuclear_feat = self.nuclare_emb(z[start_index:end_index])
            electronic_feat = self.elec_emb(self.elec[z[start_index:end_index], :])
            
            atom_features = [get_atom_features(atom) for atom in mol_with_h.GetAtoms()]
            atom_features = torch.tensor(atom_features, dtype=torch.float32).to(self.device)
            atom_supp = self.atom_supp_emb(atom_features)
            
            cls_vector = self.json_data[smile]['cls_repr']
            if isinstance(cls_vector, list):
                cls_vector = torch.tensor(cls_vector, dtype=torch.float32)
            cls_vector = cls_vector.to(self.device)
            
            # Apply attention to cls_vector
            cls_vector_transformed = self.cls_attention(cls_vector.unsqueeze(0))
            
            # Generate molecular fingerprint
            fingerprint = self.get_molecular_fingerprints(mol)
            fingerprint = torch.tensor(fingerprint, dtype=torch.float32).to(self.device)
            
            # Apply dynamic weight attention to fingerprint and cls_vector
            cls_fusion, attention_weights = self.dynamic_attention(fingerprint.unsqueeze(0), cls_vector.unsqueeze(0))
            attention_weights_list.append(attention_weights)
            
            # Repeat vectors for each atom in the molecule
            cls_vector = cls_vector.repeat(smiles_with_h_count, 1)
            cls_vector_transformed = cls_vector_transformed.repeat(smiles_with_h_count, 1)
            cls_fusion = cls_fusion.repeat(smiles_with_h_count, 1)
            
            current_feat = nuclear_feat + electronic_feat + 0.1 * atom_supp + cls_vector + 0.2 * cls_vector_transformed + 0.2 * cls_fusion
            
            total_feat[start_index:end_index, :] = current_feat
            start_index = end_index
        
        S = self.act(self.ls(total_feat)).to(self.device)
        return S
