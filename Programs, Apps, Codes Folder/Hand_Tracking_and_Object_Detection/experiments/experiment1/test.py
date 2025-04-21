import os
from pathlib import Path

def validate_yolo(model_path, dataset_config, img_size=640, project="models/yolo", name="validation"):
    """
    执行YOLOv8模型验证
    :param model_path: 训练好的模型路径（best.pt）
    :param dataset_config: 数据集配置文件路径
    :param img_size: 输入图片尺寸（需与训练一致）
    :param project: 验证结果保存目录
    :param name: 验证实验名称
    """
    # 构建验证命令
    val_cmd = (
        f"yolo task=detect mode=val "
        f"model={model_path} "
        f"data={dataset_config} "
        f"imgsz={img_size} "
        f"project={project} "
        f"name={name} "
        f"batch=16 "  # 根据GPU显存调整
        f"device=0 "  # 使用GPU
        f"save_json=True "  # 保存JSON结果
        f"plots=True"  # 生成评估图表
    )
    
    print(f"执行验证命令: {val_cmd}")
    os.system(val_cmd)

def test():
    """
    批量验证所有训练配置
    :param base_model: 基础模型名称（用于构建路径）
    """
    # 定义实验配置列表
    experiments = [
        {
            "name": "yolo_all",
            "dataset": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_all/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_all/weights/best.pt"
        },
        {
            "name": "yolo_dataset_0_1",
            "dataset":os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_0_1/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_0_1/weights/best.pt"
        },
        {
            "name": "yolo_dataset_1_2",
            "dataset":os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_1_2/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_1_2/weights/best.pt"
        },
        {
            "name": "yolo_dataset_2_3",
            "dataset":os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_2_3/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_2_3/weights/best.pt"
        },
        {
            "name": "yolo_dataset_3_4",
            "dataset":os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_3_4/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_3_4/weights/best.pt"
        },
        {
            "name": "yolo_dataset_4_5",
            "dataset":os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/ultralytics/datasets/yolo_dataset_4_5/data.yaml")),
            "model_dir": f"models/yolo/yolo_dataset_4_5/weights/best.pt"
        }
    ]

    for exp in experiments:
        print(f"\n{'='*30} 正在验证实验 {exp['name']} {'='*30}")
        
        # 检查模型文件是否存在
        if not Path(exp["model_dir"]).exists():
            print(f"警告：未找到模型文件 {exp['model_dir']}")
            continue
            
        # 执行验证
        validate_yolo(
            model_path=exp["model_dir"],
            dataset_config=exp["dataset"],
            name=f"{exp['name']}_validation",  # 验证结果单独保存
            img_size=640
        )
