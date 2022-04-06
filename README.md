# TensorRTBase
TensorRT的封装类，支持序列化引擎，读取，存储等
<br>封装类对应的声明文件在common/TRTModelBase.h中
<br>子类继承父类，需重写processInput和processOutput方法。
<br>processInput读取数据，如图片点云，并将其存到buffers缓存中。
<br>processOutput读取数据，将最后的输出结果，如one-hot编码处理为自己需要的格式
