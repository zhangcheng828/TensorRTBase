# TensorRTBase
封装类有着下列方法实现

## ModelType枚举类
分别对应onnx文件和caffe文件，本项目都支持TRT转化。

## 私有方法
### build(); 
创建network，解析onnx/caffe文件，创建引擎
### buildFromSerializedEngine(); 
反序列化TRT引擎，需提供TRT引擎的存储路径
### loadEngine(); 
加载TRT引擎
### engineInitlization(); 
若mParams.load_engine为真，则从文件路径反序列化TRT文件，调用buildFromSerializedEngine方法，若为假，调用build方法，需重新解析引擎。

## 受保护的方法（需要重写的接口）
### processInput(); 
读取数据，如图片点云，并将其存到buffers缓存中。此接口需要子类重写。
### processInput(); 
将最后的输出结果，如one-hot编码处理为自己需要的格式。此接口需要子类重写。

## 公有方法
### TRTModelBase构造函数
直接调用engineInitlization方法
### forward();
模拟神经网络的前向传播
### saveEngine();
保存序列化引擎
