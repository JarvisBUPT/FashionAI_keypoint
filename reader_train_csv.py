import csv
from itertools import islice

data_path = "/home/sk39/workspace/dataset/tianchi_clothes/train/Annotations/annotations.csv"
test_path = "test.csv"
f = open(test_path, "r")
keys = f.readline()  # 读取第一行的内容
print(keys)
f.close()
imgs_dict = {}
data_table = []
with open(test_path, "r") as f:
    for value in islice(f, 1, None):
        print(value)
        imgs_dict[keys[i]] = v
# with open(test_path, "r") as f:
#     reader = csv.DictReader(f)
#     rows = [row for row in reader]
print(rows)
