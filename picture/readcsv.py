import csv
import os


def readresult(path=None):
    """

    Args:
        path: A str, path of csv file, if is None, set path to result.csv

    Returns:

    """
    if path is None:
        path = 'result.csv'

    cwd = os.getcwd()  # 获取该文件工作空间
    dir_name = os.path.dirname(__file__)  # 获取文件所在目录
    abs_path = os.path.abspath(__file__)  # 获取文件所在目录加该文件名，含后缀
    ch_dir = os.chdir(os.path.dirname(__file__))  # 修改该文件的工作空间为文件所在目录
    parent_path = os.path.dirname(dir_name)  # 获得d所在的目录,即d的父级目录
    ch_dir_parent = os.chdir(parent_path)
    parent2_path = os.path.dirname(parent_path)  ##获得parent_path所在的目录即parent_path的父级目录

    print('cwd', cwd)
    print('dir_name', dir_name)
    print('abs_path', abs_path)
    print('ch_dir', ch_dir)
    print('parent_path', parent_path)
    print('parent2_path', parent2_path)
    # with open(path, 'r', newline='') as f:
    #     reader = csv.reader(f)
    #     for v in reader:
    #         print(v)
