import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import psutil
import gc
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

from stm_yolo import MultiViewVideoClassifier
from yolov8n_cls import YOLOBaselineClassifier  # 导入基线模型
from dataset import MultiViewVideoDataset
from ultralytics.utils.loss import v8ClassificationLoss


class Config:
    ORIGINAL_DATA_DIR = r'C:\Users\11951\Desktop\NUS\STM_YOLO\stm_yolo\stmyolo_dataset_csv'
    PROCESSED_DATA_DIR = r'C:\Users\11951\Desktop\NUS\STM_YOLO\stm_yolo\stmyolo_dataset_frames'
    ALL_STAGES = [f'stage_{i}' for i in range(5)]
    TRAIN_VIDEOS = [f'video_{i+1}' for i in range(12)]
    TEST_VIDEOS = [f'video_{i+1}' for i in range(12, 15)]
    NUM_CLASSES = 2
    SEQUENCE_LENGTH = 5
    IMG_SIZE = 224
    EPOCHS = 30
    BATCH_SIZE = 12
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASELINE_PREFIX = 'baseline'
    STM_PREFIX = 'stm'


def save_metrics(report_dict, model_prefix, stage_name):
    file_path = f"{model_prefix}_{stage_name}_metrics.csv"
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in report_dict.items():
            writer.writerow([key, value])
    print(f"✅ 测试指标结果保存至 {file_path}")


def get_memory_info(device):
    cpu_usage = psutil.virtual_memory().percent
    cpu_allocated = psutil.virtual_memory().used / (1024**3)
    info_str = f"CPU Mem: {cpu_usage:.1f}% ({cpu_allocated:.2f} GB)"
    if device.type == 'cuda':
        gpu_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        gpu_cached = torch.cuda.memory_reserved(device) / (1024**3)
        info_str += f" | GPU Mem: Allocated={gpu_allocated:.2f} GB, Cached={gpu_cached:.2f} GB"
    return info_str


def train_one_epoch(model, dataloader, criterion, optimizer, device, is_baseline=False):
    model.train()
    running_loss = 0.0
    for view1, view2, view3, labels in tqdm(dataloader, desc="训练", unit="batch"):
        view1, view2, view3, labels = view1.to(device), view2.to(device), view3.to(device), labels.to(device)
        optimizer.zero_grad()
        if is_baseline:
            outputs = model(view1, view2, view3)  # baseline 仅用 view3 的最后一帧
        else:
            outputs = model(view1, view2, view3)
        loss, _ = criterion(outputs, {"cls": labels})
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        del view1, view2, view3, labels, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    return running_loss / len(dataloader)


def test(model, dataloader, device, num_classes, is_baseline=False):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for view1, view2, view3, labels in tqdm(dataloader, desc="测试", unit="batch"):
            view1, view2, view3, labels = view1.to(device), view2.to(device), view3.to(device), labels.to(device)
            outputs = model(view1, view2, view3)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0) * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, zero_division=0)
    print(f"\n📊 测试集结果:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    return {
        "Accuracy (%)": accuracy,
        "Precision (%)": precision,
        "Recall (%)": recall,
        "F1 Score (%)": f1
    }


def save_loss_curve_csv(losses, model_prefix, stage_name):
    csv_path = f"{model_prefix}_{stage_name}_loss_curve.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train_Loss"])
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, loss])
    print(f"📉 训练损失曲线保存至 {csv_path}")


def run_training_pipeline(model_class, model_prefix, cfg: Config, train_loader, test_loader, device, is_baseline=False):
    model = model_class(num_classes=cfg.NUM_CLASSES).to(device)
    criterion = v8ClassificationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    train_losses = []
    for epoch in range(cfg.EPOCHS):
        print(f"\n--- [{model_prefix}] Epoch {epoch + 1}/{cfg.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, is_baseline)
        print(f"训练损失: {train_loss:.4f}")
        train_losses.append(train_loss)
        torch.save(model.state_dict(), f"{model_prefix}_{cfg.STAGE_FOLDER}_best_model.pth")

    save_loss_curve_csv(train_losses, model_prefix, cfg.STAGE_FOLDER)
    print("开始测试...")
    results = test(model, test_loader, device, cfg.NUM_CLASSES, is_baseline)
    save_metrics(results, model_prefix, cfg.STAGE_FOLDER)


def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)

    for stage_folder in cfg.ALL_STAGES:
        cfg.STAGE_FOLDER = stage_folder
        print(f"\n======== 📦 开始训练阶段: {cfg.STAGE_FOLDER} ========")
        frames_stage_path = os.path.join(cfg.PROCESSED_DATA_DIR, cfg.STAGE_FOLDER)
        labels_stage_path = os.path.join(cfg.ORIGINAL_DATA_DIR, cfg.STAGE_FOLDER)

        train_dataset = MultiViewVideoDataset(frames_stage_dir=frames_stage_path, labels_stage_dir=labels_stage_path,
                                              sequence_length=cfg.SEQUENCE_LENGTH, transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), antialias=True),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
                                              ]), video_folders=cfg.TRAIN_VIDEOS)
        test_dataset = MultiViewVideoDataset(frames_stage_dir=frames_stage_path, labels_stage_dir=labels_stage_path,
                                             sequence_length=cfg.SEQUENCE_LENGTH, transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), antialias=True),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                             ]), video_folders=cfg.TEST_VIDEOS)

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        #run_training_pipeline(MultiViewVideoClassifier, cfg.STM_PREFIX, cfg, train_loader, test_loader, device, is_baseline=False)
        run_training_pipeline(YOLOBaselineClassifier, cfg.BASELINE_PREFIX, cfg, train_loader, test_loader, device, is_baseline=True)


if __name__ == '__main__':
    main()

