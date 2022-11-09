import torch
import os

def save_encoder_and_decoder(model_path,encoder_path,decoder_path):
    model = torch.load(model_path)
    encoder = model.encoder
    decoder = model.decoder
    torch.save(encoder,encoder_path)
    torch.save(decoder,decoder_path)

if __name__ ==  "__main__":
    if not os.path.exists("output_xcoder"):
        os.mkdir("output_xcoder")
    model_path = "output_train/g2pE_mobile_best.pth"
    encoder_path = "output_xcoder/g2pE_mobile_encoder.pth"
    decoder_path = "output_xcoder/g2pE_mobile_decoder.pth"
    save_encoder_and_decoder(model_path,encoder_path,decoder_path)
