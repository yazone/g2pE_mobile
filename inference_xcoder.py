import torch
import numpy as np
from config import Config

def get_input_tokens(inp, config):
    tokens = list(inp) + [Config.G2P_FLAG_END]
    x = [config.g2idx.get(t, config.g2idx[Config.G2P_FLAG_UNK]) for t in tokens]
    return x

g2pE_encoder = None
g2pE_decoder = None

def get_encoder():
    global g2pE_encoder
    g2pE_encoder = torch.load("output_xcoder/g2pE_mobile_encoder.pth")
    return g2pE_encoder
    
def get_decoder():
    global g2pE_decoder
    g2pE_decoder = torch.load("output_xcoder/g2pE_mobile_decoder.pth")
    return g2pE_decoder

def predict_encoder(input_text,device,config):
    input_text = input_text.lower()
    input_tokens = get_input_tokens(input_text,config)
    
    seqlens = np.array([len(input_tokens)])
    input_tokens = np.array([input_tokens])
    
    input_tokens = torch.as_tensor(input_tokens).to(device)
    seqlens = torch.as_tensor(seqlens)
    
    encoder = get_encoder()
    encoder.to(device)
    print("input_tokens shape:",input_tokens.shape,"seqlens shape:",seqlens.shape)
    encoder_output = encoder(input_tokens, seqlens)
    return encoder_output

def predict_decoder(h0,device,config):
    decoder = get_decoder()
    decoder.to(device)
    decoder_inputs = torch.as_tensor(np.array([[2]])).to(device)
    logits, y_hat = [], []
    for t in range(20):
        _logits, _y_hat, h0 = decoder(decoder_inputs, h0)
        if config.idx2p[_y_hat.cpu().numpy()[0][0]] == Config.G2P_FLAG_END:
            break
        logits.append(_logits)
        y_hat.append(_y_hat)
        decoder_inputs = _y_hat

    logits = torch.cat(logits, 1)
    y_hat = torch.cat(y_hat, 1) 
    return logits,y_hat
    
def predict(input_text,config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_output = predict_encoder(input_text,device,config)
    print("encoder_output shape:",encoder_output.shape)
    encoder_output = encoder_output.unsqueeze(0)
    print("encoder_output shape:",encoder_output.shape)
    logits,y = predict_decoder(encoder_output,device,config)
    
    y = y[0].cpu().numpy()
    y_phoneme = []
    for idx in y:
        y_phoneme.append(config.idx2p[idx])
    return y_phoneme
    
config = Config()    
y_phoneme = predict("CARDER",config)
# K AA1 R D ER0
print("predict result is:"," ".join(y_phoneme))

