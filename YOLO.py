import torch
from ultralytics import YOLO
def main():
    model = YOLO(".pt")
    results = model.train(
        data="",
        epochs=150,
        imgsz=224,
        project="runs/classify",
        name="exp1",
        lr0=0.001,
        batch=32,
        workers=0
    )
    print("result：")
    print(f"train loss：{results.loss[0]}")
    print(f"train acc：{results.map50}")
    print("Val result：")
    print(f"val loss：{results.val_loss}")
    print(f"val acc：{results.val_map50}")
if __name__ == '__main__':
    main()


