from ultralytics import YOLO
import cv2
import os
import csv

def load_best_model():
    # 加载最佳训练模型
    model = YOLO("")  # 加载训练好的模型权重文件
    return model

def predict_image(model, image_path):
    # 使用 OpenCV 加载图片
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 进行预测
    results = model(img_rgb)

    # 假设结果是列表，获取第一个结果
    if isinstance(results, list):
        results = results[0]

    # 获取预测的类别名
    class_names = results.names

    # 处理预测的概率信息
    if hasattr(results, 'probs'):
        probs = results.probs
        # 获取 top1 类别的索引和概率
        class_index = probs.top1
        max_prob = probs.top1conf.item()

        predicted_class = class_names[class_index]
        confidence_score = max_prob
    else:
        predicted_class = "N/A"
        confidence_score = 0.0

    return image_path, predicted_class, confidence_score

def main():
    # 加载最佳训练模型
    model = load_best_model()

    # 指定要预测的图片文件夹路径
    image_folder = ''

    # 获取文件夹中的所有图片文件路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 打开一个CSV文件来写入预测结果
    with open('prediction_results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入CSV文件的表头
        writer.writerow(['Image Path', 'Predicted Class', 'Confidence Score'])

        # 遍历并预测每张图片
        for image_path in image_paths:
            print(f"Predicting for {image_path}")
            image_path, predicted_class, confidence_score = predict_image(model, image_path)
            # 写入每张图片的预测结果到CSV文件
            writer.writerow([image_path, predicted_class, confidence_score])

if __name__ == '__main__':
    main()
