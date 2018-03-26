"""
TRAIN LAUNCHER

"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from datagenclothes import DataGenClothes
from model_hourglass import HourglassModelForClothes
import os
import sys
from processconfig import process_config_clothes




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
    dataset = DataGenClothes(params['joint_list'], params['img_directory'], params['training_txt_file'],
                             category, cat)
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()
    name = params['name'] + cat
    model = HourglassModelForClothes(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                                     nLow=params['nlow'], outputDim=params['num_joints'],
                                     batch_size=params['batch_size'],
                                     attention=params['mcam'], training=True, drop_rate=params['dropout_rate'],
                                     lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],
                                     decay_step=params['decay_step'], dataset=dataset, name=name,
                                     logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],
                                     tiny=params['tiny'], w_loss=params['weighted_loss'], joints=params['joint_list'],
                                     modif=False)
    model.generate_model()
    model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
                        dataset=None)
