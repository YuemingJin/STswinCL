import torch
import torch.nn as nn
from collections import OrderedDict


def load_model_cata(model, pretrain_dir, log=True):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    # convert data_parallal to model
    # for key in state_dict_:
    #     if key.startswith('module') and not key.startswith('module_list'):
    #         state_dict[key[7:]] = state_dict_[key]
    #     else:
    #         state_dict[key] = state_dict_[key]

    # convert data_parallal to model & only read resnet part
    for key in state_dict_:
        print('load key:', key)
        if key.startswith('module.resnet') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

def load_model(model, pretrain_dir, log=True):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    # convert data_parallal to model
    # for key in state_dict_:
    #     if key.startswith('module') and not key.startswith('module_list'):
    #         state_dict[key[7:]] = state_dict_[key]
    #     else:
    #         state_dict[key] = state_dict_[key]

    # convert data_parallal to model & only read resnet part
    for key in state_dict_:
        #print('!!load key:',key)
        if key.startswith('resnet'):
            state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    # for key in model_state_dict:
    #     print('model key!!',key)
    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model


def load_model_full(model, pretrain_dir, log=True):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    # convert data_parallal to model
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model

def load_model_test(model, pretrain_dir, log=True):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    # convert data_parallal to model
    for key in state_dict_:
        #print('!!load key:', key)
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    # for key in model_state_dict:
    #     print('model key!!',key)

    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model

def load_model_CL(model, pretrain_dir, log=True):


    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')

    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    for key in state_dict_['model']:
        if key.startswith('pixpro.encoder_1'):
            state_dict['resnet'+key[16:]] = state_dict_['model'][key]
        # elif key.startswith('pixpro.encoder_1.layer'):
        #     state_dict2['resnet'+key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_2'):
            state_dict['memory' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_3'):
            state_dict['aspp' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.proj.'):
            state_dict['project' + key[11:]] = state_dict_['model'][key]


  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model

def load_model_swin_CL(model, pretrain_dir, log=True):


    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')

    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    for key in state_dict_['model']:
        if key.startswith('pixpro.encoder_1'):
            state_dict['resnet'+key[16:]] = state_dict_['model'][key]
        # elif key.startswith('pixpro.encoder_1.layer'):
        #     state_dict2['resnet'+key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_2'):
            state_dict['swin' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_3'):
            state_dict['aspp' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.proj.'):
            state_dict['project' + key[11:]] = state_dict_['model'][key]


  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model

def load_model_mswin_CL(model, pretrain_dir, log=True):


    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')

    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    for key in state_dict_['model']:
        if key.startswith('pixpro.encoder_1'):
            state_dict['resnet'+key[16:]] = state_dict_['model'][key]
        # elif key.startswith('pixpro.encoder_1.layer'):
        #     state_dict2['resnet'+key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_2'):
            state_dict['swin' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.encoder_3'):
            state_dict['aspp' + key[16:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.proj1'):
            state_dict['project1' + key[12:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.proj2'):
            state_dict['project2' + key[12:]] = state_dict_['model'][key]
        elif key.startswith('pixpro.proj3'):
            state_dict['project3' + key[12:]] = state_dict_['model'][key]

  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model