import torch
import numpy as np

from config import Config


def get_input_tokens(input_text,config):
    input_text = input_text.lower()
    tokens = list(input_text) + [config.G2P_FLAG_END]
    input_tokens = [config.g2idx.get(t, config.g2idx[config.G2P_FLAG_UNK]) for t in tokens]
    return input_tokens

if __name__ == "__main__":
    config = Config()
    
    # result should be "K AA1 R D ER0"
    # set input
    input_text = 'CARDER'
    input_tokens = get_input_tokens(input_text,config) # input tokens should be "5,  3, 20,  6,  7, 20,  2"
    decoder_input = config.p2idx[config.G2P_FLAG_START]
    print("input_text:",input_text)

    # use numpy
    seqlens = np.array([len(input_tokens)])
    input_tokens = np.array([input_tokens])
    decoder_input = np.array([[decoder_input]])
    print("numpy input_tokens:",input_tokens,"seqlens:",seqlens,"decoder_input:",decoder_input)

    # set input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tokens = torch.as_tensor(input_tokens).to(device)
    seqlens = torch.as_tensor(seqlens)
    decoder_input = torch.as_tensor(decoder_input).to(device)
    print("tensor input_tokens:",input_tokens,"seqlens:",seqlens,"decoder_input:",decoder_input)

    # run model
    model = torch.load("output_train/g2pE_mobile_best.pth")
    model.eval()
    model.to(device)
    _,y = model(input_tokens,seqlens,decoder_input,teacher_forcing=False,dec_maxlen=20)
    print("predict y:",y)
    
    # print result
    y_phoneme = []
    y = y[0].cpu().numpy()
    for idx in y:
        # end flag
        if config.idx2p[idx] == config.G2P_FLAG_END:
            break
        y_phoneme.append(config.idx2p[idx])
    print("predict result is:"," ".join(y_phoneme))


