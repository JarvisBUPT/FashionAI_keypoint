import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from datagenclothes import DataGenClothes
from model_hourglass import HourglassModelForClothes
import os
import sys
import platform


def process_config_clothes():
    """
    """
    machine_name = platform.node()
    if machine_name == 'P100v0':
        conf_file = 'config_clothes.cfg'
    elif machine_name == 'Jason':
        conf_file = 'config_clothes_win.cfg'
    elif machine_name == 'localhost.localdomain':
        conf_file = 'config_hcy.cfg'
    elif machine_name == 'DESKTOP-3IQHBMV':
        conf_file = 'config_clothes_win.cfg'
    else:
        conf_file = 'config_clothes.cfg'
    params = {}
    print(conf_file)
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
    print(platform.node())
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
