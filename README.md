# 简介

本项目一共提供了2个后端实现

- [x] 1. 算术验证码识别（captcha）
- [x] 2. 物体检测，用来做点选验证码识别（objec_detection）

## 算术验证码识别

1. 使用[dddd_trainer](https://github.com/JaysonAlbert/dddd_trainer.git)训练算术验证码识别模型，并导出成onnx格式
2. 使用ddddocr来识别算术验证码

## 物体检测（点选验证码）

1. 使用ddddocr检测物体框
2. 训练孪生网络[Siamese-pytorch](https://github.com/JaysonAlbert/Siamese-pytorch.git)来判断图标相似性，来进行点选识别
3. `python serve.py 启动服务`
