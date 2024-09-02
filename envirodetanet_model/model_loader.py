import torch
from .envirodetanet import EnviroDetaNet, oriDetaNet

def scalar_model(device, params='trained_param/qm7x/energy.pth', model_type='EnviroDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def force_model(device, params='trained_param/qm7x/force.pth', model_type='EnviroDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type='force',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def charge_model(device, params='trained_param/qm9spectra/npacharge.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out=None,
                 summation=False,
                 norm=False,
                 out_type='scalar',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dipole_model(device, params='trained_param/qm9spectra/dipole.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=True,
                 norm=False,
                 out_type='dipole',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def polar_model(device, params='trained_param/qm9spectra/polar.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def quadrupole_model(device, params='trained_param/qm9spectra/quadrupole.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def hyperpolar_model(device, params='trained_param/qm9spectra/hyperpolar.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def octapole_model(device, params='trained_param/qm9spectra/octapole.pth', model_type='EnviroDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hi_model(device, params='trained_param/qm9spectra/Hi.pth', model_type='oriDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hi',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hij_model(device, params='trained_param/qm9spectra/Hij.pth', model_type='oriDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hij',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dedipole_model(device, params='trained_param/qm9spectra/dedipole.pth', model_type='oriDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=False,
                 norm=False,
                 out_type='dipole',
                 grad_type='dipole',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def depolar_model(device, params='trained_param/qm9spectra/depolar.pth', model_type='oriDetaNet'):
    state_dict=torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=max_number,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=False,
                 norm=False,
                 out_type='2_tensor',
                 grad_type='polar',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def nmr_model(device, params, model_type='EnviroDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def uv_model(device, params='trained_param/qm9spectra/borden_os.pth', model_type='EnviroDetaNet'):
    state_dict = torch.load(params)
    ModelClass = EnviroDetaNet if model_type == 'EnviroDetaNet' else oriDetaNet
    max_number = 35 if model_type == 'EnviroDetaNet' else 9
    model = ModelClass(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=240,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model