
from torch.utils import data

class G2pDataset(data.Dataset):
    def __init__(self, words, prons, config):
        """
        words: list of words. e.g., ["w o r d", ]
        prons: list of prons. e.g., ['W ER1 D',]
        """
        self.words = words
        self.prons = prons
        self.config = config

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word, pron = self.words[idx], self.prons[idx]
        x = self.encode(word, "x", self.config.g2idx)
        y = self.encode(pron, "y", self.config.p2idx)
        decoder_input, y = y[:-1], y[1:]
        x_seqlen, y_seqlen = len(x), len(y)
                
        return x, x_seqlen, word, decoder_input, y, y_seqlen, pron
        
    def encode(self, input_text, input_type, idx_dict):
        '''convert string into ids
        type: "x" or "y"
        dict: g2idx for 'x', p2idx for 'y'
        '''
        if input_type == "x": 
            tokens = input_text.split() + ["</s>"]
        else: 
            tokens = ["<s>"] + input_text.split() + ["</s>"]

        x = [idx_dict.get(t, idx_dict["<unk>"]) for t in tokens]
        
        return x
    
