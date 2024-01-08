from models.eegnet import EEGNet
from models.tfr_net import TFRNet
from models.lstm import LSTMClassifier
from models.rnn import RNNClassifier
from models.attention import AttentionEEGNet


def load(model_name: str, **kwargs):
    if model_name == 'eegnet':
        return EEGNet(**kwargs)
    elif model_name == 'tfrnet':
        return TFRNet()
    elif model_name == 'lstm':
        return LSTMClassifier(**kwargs)
    elif model_name == 'rnn':
        return RNNClassifier(**kwargs)
    elif model_name == 'eegnet_attention':
        return AttentionEEGNet(**kwargs)
