# g2pE_mobile: A Simple Neural Network For English Grapheme To Phoneme Conversion(Just For OOV)

模型代码参考自： [g2p](https://github.com/Kyubyong/g2p)

## （一）架构
1. 训练&推理：pytorch+MNN；
2. 算法：encoder+decoder生成模型；

## （二）功能
1. 训练英文g2p模型，用于处理TTS中OOV问题；
2. 转换成onnx->mnn，用于离线推理；

## （三）训练&评估&推理
### 1. 训练模型
```
python train.py
```
### 2. 评估模型
修改train.py
```
if __name__ == "__main__":
    # set True for eval only,set False for train
    do_train(True)    
````
### 3. 模型推理
```
python inference.py
```

## （四）Encoder、Decoder推理
当解码器遇到结束符时需要提前结束解码，在不修改先前模型基础上将Encoder、Decoder模块分别保存进行推理。
### 1. 保存Encoder、Decoder模型
```
python export_xcoder.py
```
### 2. 使用Encoder、Decoder推理
```
python inference_xcoder.py
```

## （五）onnx中间格式
### 1. 转换onnx模型
```
python export_xcoder.py
```
### 2. 使用onnx推理
（没做）


## （六）移动端MNN推理
### 1. onnx转换mnn模型
执行sh，注意脚本中修改MNNConvert路径
```
./export_mnn.sh
```
### 2. 使用mnn推理
模型加载与推理已经在demo_mnn/G2pEModel.cpp中实现，在C++代码中将类实例化进行推理，实测四核A35单个单词耗时毫秒级。
