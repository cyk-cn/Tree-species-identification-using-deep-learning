import torch
from ultralytics import YOLO

def main():
    # 加载分类模型
    model = YOLO("yolo11x-cls.pt")  # 或者你自己下载的 yolov11x-cls.pt

    # 开始训练
    results = model.train(
        data="D:\project-yolo\dataset_split\无人机-1",  # 根目录，包含 train/ 和 val/ 子目录
        epochs=150,
        imgsz=224,
        project="runs/classify",
        name="exp1",
        lr0=0.001,
        batch=32,
        workers=0  # 禁用多进程数据加载（如果遇到多进程问题）
    )

    # 输出训练集和验证集的损失函数和准确率变化
    print("训练结果：")
    print(f"训练集损失：{results.loss[0]}")  # 损失
    print(f"训练集准确率：{results.map50}")  # mAP@50

    # 输出验证集损失和准确率
    print("验证结果：")
    print(f"验证集损失：{results.val_loss}")  # 验证集损失
    print(f"验证集准确率：{results.val_map50}")  # 验证集mAP@50

if __name__ == '__main__':
    main()


