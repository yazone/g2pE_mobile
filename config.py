class Config:
    # model config
    batch_size = 256
    enc_maxlen = 20
    dec_maxlen = 20
    num_epochs = 2
    hidden_units = 64
    emb_units = 64
    G2P_FLAG_PAD = "<pad>"
    G2P_FLAG_UNK = "<unk>"
    G2P_FLAG_START = "<s>"
    G2P_FLAG_END = "</s>"
    graphemes = [G2P_FLAG_PAD, G2P_FLAG_UNK, G2P_FLAG_END] + list("abcdefghijklmnopqrstuvwxyz")
    phonemes = [G2P_FLAG_PAD, G2P_FLAG_UNK, G2P_FLAG_START, G2P_FLAG_END] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
                    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
                    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW',
                    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
    # train config
    lr = 0.001
    logdir = "log/01"
    
    def __init__(self):
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
