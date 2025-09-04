from models.edsr import EDSR
import torch
from types import SimpleNamespace
from models.wavemixsr import WaveMixSR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_edsr( scale=4):
    '''Instantiates the EDSR model with the given scale factor and loads pre-trained weights.'''
    args = SimpleNamespace()
    args.scale = [scale]
    args.n_resblocks = 32
    args.n_feats = 256
    args.rgb_range = 255
    args.n_colors = 3
    args.res_scale = 0.1
    args.no_upsampling = False
    args.act = 'relu'

    model = EDSR(args).to(DEVICE)

    ckpt = torch.load('path/to/edsr/model', map_location=DEVICE)
    model.load_state_dict(ckpt)
    #model.eval()
    return model


def instantiate_wavemixsr(scale=4):
    '''Instantiates the WaveMixSR model and loads pre-trained weights.'''
    model = WaveMixSR(
        depth = 4,
        mult = 1,
        ff_channel = 144,
        final_dim = 144,
        dropout = 0.3
    )
    state_dict = torch.load(f'path/to/wavemixsr/model',map_location=DEVICE)
    model.load_state_dict(state_dict)
    #model.eval()
    return model
    
def get_model(model_name, scale=4):
    if model_name == 'edsr':
        return instantiate_edsr(scale)
    elif model_name == 'wavemixsr':
        return instantiate_wavemixsr(scale)
    else:
        raise ValueError(f"Model {model_name} is not supported.")