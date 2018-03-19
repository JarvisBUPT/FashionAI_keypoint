import csv
from itertools import islice

"""
    该文件是一个测试文件，测试怎么使用读取和写入csv文件
"""
data_path = "/home/sk39/workspace/dataset/tianchi_clothes/train/Annotations/annotations.csv"
test_path = "test.csv"
f = open(test_path, "r")
keys = f.readline().split(',')  # 读取第一行的内容
a = False
b= True
if not a and b:
    print("1")
# print(keys)
# f.close()
# imgs_dict = {}
# data_table = []
# with open(test_path, "r") as f:
#     for value in islice(f, 1, None):
#         v = value.split(',')
#         print(v)

# with open(test_path, "r") as f:
#     reader = csv.DictReader(f)
#     rows = [row for row in reader]
# print(rows)
