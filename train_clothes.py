"""
TRAIN LAUNCHER

"""

import sys

from configs.processconfig import process_config_clothes
from datagen.datagenclothes import DataGenClothes
from models.model_hourglass import HourglassModelForClothes

if __name__ == '__main__':
    print('--Parsing Config File')
    argv = sys.argv
    if len(argv) == 2:
        c = argv[1]
    else:
        c = ''
    params = process_config_clothes()
    category = []
    if c == 'b':
        category.append('blouse')
        cat = 'blouse'
    elif c == 'd':
        category.append('dress')
        cat = 'dress'
    elif c == 'o':
        category.append('outwear')
        cat = 'outwear'
    elif c == 's':
        category.append('skirt')
        cat = 'skirt'
    elif c == 't':
        category.append('trousers')
        cat = 'trousers'
    else:
        category = params['category']
        cat = ''
    print('categoty =', category, cat)

    name = params['name'] + cat  # params['name']=hg_clothes_001+'blouse'
    if cat == '':
        num_joints = 24
    else:
        num_joints = len(params[cat])
    joint_list = params['joint_list']
    joints = []
    if cat == '':
        joints = joint_list
    else:
        for i, v in enumerate(joint_list):
            if i in params[cat]:
                joints.append(v)
    print(joints)
    if cat == '':
        dataset = DataGenClothes(joints, params['img_directory'], params['training_txt_file'], category, cat)
    else:
        dataset = DataGenClothes(joints, params['img_directory'], "split_" + cat + ".csv",
                                 category, cat)
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()
    model = HourglassModelForClothes(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                                     nLow=params['nlow'], outputDim=num_joints,
                                     batch_size=params['batch_size'],
                                     attention=params['mcam'], training=True, drop_rate=params['dropout_rate'],
                                     lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],
                                     decay_step=params['decay_step'], dataset=dataset, name=name,
                                     logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],
                                     tiny=params['tiny'], w_loss=params['weighted_loss'], joints=joints,
                                     modif=False)
    model.generate_model()
    # load =  './hourglass_saver/model/' + params['name'] + '/' + params['name'] + "_" + '37'
    #load =  './hourglass_saver/model/hg_clothes_005/hg_clothes_005_69'
    load = None
    model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
                        dataset=None, load=load)
