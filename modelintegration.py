import csv
import sys
from time import strftime
from time import time

sys.path.append('..')

a = strftime('%m%d%H%M')
print(a)
if __name__ == '__main__':
    starttime = time()
    argv = sys.argv
    if len(argv) != 3:
        raise ValueError('need two parameter ,which is the number of epoch\n'
                         'for example: python modelintegration.py 01131458 04131459 ')
    file1 = argv[1]
    file2 = argv[2]
    f1 = open(file1, 'r')  # if has no the file ,will raise FileNotFound
    f2 = open(file2, 'r')
    with open('result' + strftime('%m%d%H%M') + '.csv', 'w', newline='') as f:  # if has no the file ,will create it
        writer = csv.writer(f)
        firstline = ['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front',
                     'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                     'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out',
                     'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left',
                     'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in',
                     'bottom_right_out']
        writer.writerow(firstline)
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        next(reader1)
        next(reader2)
        # for i in range(len(reade1))
        for value1, value2 in zip(reader1, reader2):
            # value1 is a list, the data is the one row of csv file, value2 is the same
            keypoint1 = value1[2:]  # the coordinate x_y_v
            keypoint2 = value2[2:]  # x_y_v
            if value1[0] == value2[0]:
                img_name = value1[0]
                cat_temp = value1[1]
                print(img_name, cat_temp)
                joints = []  # write the data row into the result.csv
                joints.append(img_name)
                joints.append(cat_temp)
                for i, j in zip(keypoint1, keypoint2):
                    coord1 = i.split('_')
                    coord2 = j.split('_')
                    # print(coord2)
                    joints.append(
                        str((int(coord1[0]) + int(coord2[0])) // 2) + '_' +
                        str((int(coord1[1]) + int(coord2[1])) // 2) + '_1')
                writer.writerow(joints)
                # img_dst = os.path.join(params['log_dir_test'], cat_temp)
                # if not os.path.exists(img_dst):
                #     os.makedirs(img_dst)
                # img_plot = pre.plt_skeleton(img, joints_plot, params['joint_list'])
                # cv2.imwrite(os.path.join(img_dst, img_name.split('/')[2]), img_plot)
    f1.close()
    f2.close()
    print('model integration has', time() - starttime, 'sec')
