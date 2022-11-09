import numpy as np
from distance import levenshtein
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Encoder,Decoder,Net
from cmudict import CMUDict
from config import Config
from dataset import G2pDataset

def prepare_data(cmu_data):
    words = []
    prons = []
    for word, pronounce in cmu_data.items():
        word = " ".join(list(word.lower()))
        words.append(word)
        prons.append(pronounce)
    
    # shuffle data
    indices = list(range(len(words)))
    from random import shuffle
    shuffle(indices)
    words = [words[idx] for idx in indices]
    prons = [prons[idx] for idx in indices]
    
    # create (train,eval,test) dataset
    num_train, num_test = int(len(words)*.8), int(len(words)*.1)
    train_words, eval_words, test_words = words[:num_train], \
                                          words[num_train:-num_test],\
                                          words[-num_test:]
    train_prons, eval_prons, test_prons = prons[:num_train], \
                                          prons[num_train:-num_test],\
                                          prons[-num_test:]
    
    return train_words, eval_words, test_words, train_prons, eval_prons, test_prons


def drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    """We only include such samples less than maxlen."""
    _words, _prons = [], []
    for w, p in zip(words, prons):
        if len(w.split()) + 1 > enc_maxlen: 
            continue
        if len(p.split()) + 1 > dec_maxlen: 
            continue # 1: 
        _words.append(w)
        _prons.append(p)
    return _words, _prons          


def pad(batch):
    '''Pads zeros such that the length of all samples in a batch is the same.'''
    f = lambda x: [sample[x] for sample in batch]
    x_seqlens = f(1)
    y_seqlens = f(5)
    words = f(2)
    prons = f(-1)
    
    x_maxlen = np.array(x_seqlens).max()
    y_maxlen = np.array(y_seqlens).max()
    
    f = lambda x, maxlen, batch: [sample[x]+[0]*(maxlen-len(sample[x])) for sample in batch]
    x = f(0, x_maxlen, batch)
    decoder_inputs = f(3, y_maxlen, batch)
    y = f(4, y_maxlen, batch)
    
    f = torch.LongTensor
    
    return f(x), x_seqlens, words, f(decoder_inputs), f(y), y_seqlens, prons


def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch
        
        x, decoder_inputs = x.to(device), decoder_inputs.to(device) 
        y = y.to(device)
        
        optimizer.zero_grad()
        logits, y_hat = model(x, x_seqlens, decoder_inputs)
        
        # calc loss
        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1) # (N*T,)
        loss = criterion(logits, y)
        loss.backward()
        
        optimizer.step()
        
        if i and i%100==0:
            print(f"step: {i}, loss: {loss.item()}")
        

def calc_per(Y_true, Y_pred):
    '''Calc phoneme error rate
    Y_true: list of predicted phoneme sequences. e.g., [["B", "L", "AA1", "K", "HH", "AW2", "S"], ...]
    Y_pred: list of ground truth phoneme sequences. e.g., [["B", "L", "AA1", "K", "HH", "AW2", "S"], ...]
    '''
    num_phonemes, num_erros = 0, 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        num_phonemes += len(y_true)
        num_erros += levenshtein(y_true, y_pred)

    per = round(num_erros / num_phonemes, 2)
    return per, num_erros,num_phonemes


def convert_ids_to_phonemes(ids, idx2p):
    phonemes = []
    for idx in ids:
        p = idx2p[idx]
        if p == Config.G2P_FLAG_END:
            break
        phonemes.append(p)
    return phonemes

        
def eval(model, iterator, device, dec_maxlen,config):
    model.eval()

    Y_true, Y_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch
            x, decoder_inputs = x.to(device), decoder_inputs.to(device) 

            _, y_hat = model(x, x_seqlens, decoder_inputs, False, dec_maxlen) # <- teacher forcing is suppressed.
            
            y = y.to('cpu').numpy().tolist()
            y_hat = y_hat.to('cpu').numpy().tolist()
            for yy, yy_hat in zip(y, y_hat):
                y_true = convert_ids_to_phonemes(yy, config.idx2p)
                y_pred = convert_ids_to_phonemes(yy_hat, config.idx2p)
                Y_true.append(y_true)
                Y_pred.append(y_pred)
    
    # calc per.
    per, num_errors,total_num = calc_per(Y_true, Y_pred)
    
    with open("output_train/result", "w") as fout:
        for y_true, y_pred in zip(Y_true, Y_pred):
            fout.write(" ".join(y_true) + "\n")
            fout.write(" ".join(y_pred) + "\n\n")
    
    return per,num_errors,total_num

            
def do_train(is_only_eval=True):
    config = Config()
    
    # prepare data
    cmu_data = CMUDict().dict()
    train_words, eval_words, test_words, train_prons, eval_prons, test_prons = prepare_data(cmu_data)
    # if the length of word is too long, we will drop it
    train_words, train_prons = drop_lengthy_samples(train_words, train_prons, config.enc_maxlen, config.dec_maxlen)
    
    # prepare Dataset and DataLoader to provide data
    train_dataset = G2pDataset(train_words, train_prons,config)
    eval_dataset = G2pDataset(eval_words, eval_prons,config)
    test_dataset = G2pDataset(test_words, test_prons,config)
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad)
    val_iter = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=pad)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=pad)

    # create model
    encoder = Encoder(len(config.graphemes),config.emb_units, config.hidden_units)
    decoder = Decoder(len(config.phonemes),config.emb_units, config.hidden_units)
    model = Net(encoder, decoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load weigths if exist
    if os.path.exists("output_train/g2pE_mobile_weights_best.pth"):
        weights = torch.load("output_train/g2pE_mobile_weights_best.pth")
        print("load model from best model:",weights.keys())
        model.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(model.parameters(), lr = config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # only for eval
    if is_only_eval:
        per,num_errors,total_num = eval(model, train_iter, device, config.dec_maxlen,config)
        print("train dataset per:%.2f" % per, "num errors:", num_errors,"total nums:",total_num)
        per,num_errors,total_num = eval(model, val_iter, device, config.dec_maxlen,config)
        print("val dataset per:%.2f" % per, "num errors:", num_errors,"total nums:",total_num)
        per,num_errors,total_num = eval(model, test_iter, device, config.dec_maxlen,config)
        print("test dataset per:%.2f" % per, "num errors:", num_errors,"total nums:",total_num)
        return
    
    if not os.path.exists("output_train"):
        os.mkdir("output_train")
        
    # start train loop    
    best_per,min_num_errors,total_num = eval(model, val_iter, device, config.dec_maxlen,config)
    print("init best per",best_per,"num error:",min_num_errors,"total nums:",total_num)
    for epoch in range(1, config.num_epochs+1):
        print(f"\nepoch: {epoch}")
        train(model, train_iter, optimizer, criterion, device)
        per,num_errors,_ = eval(model, val_iter, device, config.dec_maxlen,config)
        if num_errors < min_num_errors:
            print("best is ",(best_per,min_num_errors)," and current is",(per,num_errors),",need to save")
            torch.save(model, "output_train/g2pE_mobile_best.pth")
            torch.save(model.state_dict(), "output_train/g2pE_mobile_weights_best.pth")
            best_per = per
            min_num_errors = num_errors

    # save final model
    torch.save(model, "output_train/g2pE_mobile_final.pth")
    torch.save(model.state_dict(), "output_train/g2pE_mobile_weights_final.pth")

    # test
    eval(model, test_iter, device, config.dec_maxlen,config)

        
if __name__ == "__main__":
    # set True for eval only,set False for train
    do_train(False)        

