from experiments.experiment1.train import train
from experiments.experiment1.test import test
## 如果无需调整参数，直接运行即可，否则请修改参数后重新训练，训练代码如下
train(model="yolo11s.pt",epochs=300,img_size=640)
#test()