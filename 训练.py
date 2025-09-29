import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolo11x-cls.pt")
    results = model.train(
        data="D:\project-yolo\dataset_split\无人机-1",
        epochs=150,
        imgsz=224,
        project="runs/classify",
        name="exp1",
        lr0=0.001,
        batch=32,
        workers=0
    )
    print("训练结果：")
    print(f"训练集损失：{results.loss[0]}")
    print("验证结果：")
    print(f"验证集损失：{results.val_loss}")  # 验证集损失
    print(f"验证集准确率：{results.val_map50}")  # 验证集mAP@50
if __name__ == '__main__':
    main()


