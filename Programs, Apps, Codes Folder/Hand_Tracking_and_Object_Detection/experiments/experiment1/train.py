import os

def train_yolo(model=r"utils\ultralytics\yolov11s.pt", dataset = r"utils\ultralytics\dataset\yolo_dataset_all\data.yaml", epochs=110, img_size=640, project="models/yolo", name="yolo_all"):
    """
    训练 YOLOv8 目标检测模型
    :param model: 预训练模型（如 yolov8m.pt）
    :param dataset: 数据集配置文件路径
    :param epochs: 训练轮数
    :param img_size: 输入图片大小
    :param project: 训练结果保存的主目录
    :param name: 训练实验名称
    """
    train_cmd = f"yolo task=detect mode=train model={model} data={dataset} epochs={epochs} imgsz={img_size} project={project} name={name} batch=64"
    print(f"Executing: {train_cmd}")
    os.system(train_cmd)
    
def train(model=r"utils\ultralytics\yolov8s.pt",epochs=100,img_size=640):
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_all/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_all")
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_0_1/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_0_1")
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_1_2/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_1_2")
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_2_3/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_2_3")
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_3_4/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_3_4")
    train_yolo(model=model,dataset=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_4_5/data.yaml")),epochs=epochs,img_size=img_size,project="models/yolo", name="yolo_dataset_4_5")
