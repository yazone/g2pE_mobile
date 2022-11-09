import torch
import os
import numpy as np

def export_encoder_2_onnx(encoder_path,encoder_onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(encoder_path)
    model.eval()
    model.to(device)
    input_tensor = torch.randint(1,20,(1, 20)).to(device)
    torch.onnx.export(model, (input_tensor), encoder_onnx_path,
                    input_names = ["input_ids"],
                    output_names = ["output_h0"],
                    verbose=True,
                    opset_version=11,
                    dynamic_axes = {'input_ids' : {0 : 'batch_size',1 : 'in_width'}}
                    )

def export_decoder_2_onnx(decoder_path,decoder_onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(decoder_path)
    model.eval()
    model.to(device)
    input_tensor = torch.tensor([[3]]).to(device)
    h0_tensor = torch.randn([1,1,64]).to(device)
    torch.onnx.export(model, (input_tensor,h0_tensor), decoder_onnx_path,
                    input_names = ["input_id","input_h0"],
                    output_names = ["output_logit","output_y","output_h0"],
                    verbose=True,
                    opset_version=11,
                    dynamic_axes = {'input_id' : {0 : 'batch_size'}}
                    )

if __name__ ==  "__main__":    
    if not os.path.exists("output_onnx"):
        os.mkdir("output_onnx")
    encoder_path = "output_xcoder/g2pE_mobile_encoder.pth"
    decoder_path = "output_xcoder/g2pE_mobile_decoder.pth"
    encoder_onnx_path = "output_onnx/g2pE_mobile_encoder.onnx"
    decoder_onnx_path = "output_onnx/g2pE_mobile_decoder.onnx"
    export_encoder_2_onnx(encoder_path,encoder_onnx_path)    
    export_decoder_2_onnx(decoder_path,decoder_onnx_path)      
