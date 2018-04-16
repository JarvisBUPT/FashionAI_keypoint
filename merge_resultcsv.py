import csv

from configs.processconfig import process_config_clothes

if __name__ == '__main__':
    params = process_config_clothes()
    f = open(params['training_txt_file'], 'r')
    firstline = f.readline().strip().split(',')
    f.close()
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(firstline)
        for cat in params['category']:
            with open('result' + cat + '.csv', 'r') as f1:
                reader = csv.reader(f1)
                for v in reader:
                    print(v)
                    if v is None:
                        print('1')
                    else:
                        writer.writerow(v)
