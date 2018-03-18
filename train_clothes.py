"""
TRAIN LAUNCHER

"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from datagenclothes import DataGenClothes
from model_hourglass import HourglassModelForClothes
import os


def process_config_clothes(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params


if __name__ == '__main__':
    print('--Parsing Config File')
    name = os.name
    if name == 'nt':
        config_file = 'config_clothes_win.cfg'
    else:
        config_file = 'config_clothes.cfg'
    params = process_config_clothes(config_file)
    print("params =", params)
    print('--Creating Dataset')
    dataset = DataGenClothes(params['joint_list'], params['img_directory'], params['training_txt_file'],
                             params['category'])
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()

    model = HourglassModelForClothes(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                                     nLow=params['nlow'], outputDim=params['num_joints'],
                                     batch_size=params['batch_size'],
                                     attention=params['mcam'], training=True, drop_rate=params['dropout_rate'],
                                     lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],
                                     decay_step=params['decay_step'], dataset=dataset, name=params['name'],
                                     logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],
                                     tiny=params['tiny'], w_loss=params['weighted_loss'], joints=params['joint_list'],
                                     modif=False)
    model.generate_model()
    model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
                        dataset=None)
