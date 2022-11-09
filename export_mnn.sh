mkdir -p output_mnn
# set your MNNCovert Path!!!
MNN_COVERT=~/future/speech/TTS/MNN/build/MNNConvert
$MNN_COVERT -f ONNX --modelFile output_onnx/g2pE_mobile_encoder.onnx --MNNModel output_mnn/g2pE_mobile_encoder.mnn --bizCode biz  --keepInputFormat
$MNN_COVERT -f ONNX --modelFile output_onnx/g2pE_mobile_decoder.onnx --MNNModel output_mnn/g2pE_mobile_decoder.mnn --bizCode biz  --keepInputFormat
