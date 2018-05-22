import os
import csv
import shutil

src_path = r'D:\DeepLearning\衣服关键点测试结果\mcam_clothes_001\197\new\trousers'
dst_path = r'D:\DeepLearning\衣服关键点测试结果\mcam_clothes_001\197\oo\trousers'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
image_list = os.listdir(src_path)
print(image_list)
csv_path = r'new_test.csv'
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for v in reader:
        name = v[0]
        img_name = name.split('/')[-1]
        if img_name in image_list:
            shutil.move(os.path.join(src_path, img_name), dst_path)
