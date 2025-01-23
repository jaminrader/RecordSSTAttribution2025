import numpy as np

def get_settings(exp_name):

    settings_dict = {
        'main' : {
            # Data File
            'data_fn' : 'standard_tos_annual.npz',#'standardall_annual.npz', #'standard_tos_annual.npz',
            'obs_fn' : 'ersstv5_sst_annual_regrid.nc',
            # Split Info
            'split_members': (list(range(17)), list(range(17,25)), list(range(25))),
            # Mask option
            'mask_option' : None,
            # Data augmentation, options: None, 'bootstrap'
            'data_aug_method' : None,
            # Forced altering, options: None,
            'alter_forced_method' : 'shuffle_between_gridpoints_each_model_val_internal',
            'alter_input_method' : None,
            # Standardization method # 'self_by_member'
            'in_stand_method' : 'stand_self_by_member_gridpoint_stand',
            'out_stand_method' : 'gridpoint_stand',

            # Network specs
            "learn_rate" : .001, #.0001
            "patience"      : 100,
            "batch_size"    : 64,
            "max_epochs"    : 5,
            
            # Architecture specs
            "encoding_nodes" : [100, 100], # [1000,1000,],
            "code_nodes"     : 100, # 100,
            "activation"     : "tanh",# "tanh",
            "ridge" : 0.,
            "variational" : False,
            "variational_loss": 0.0001,
            "bias_init": "random", #zeros, random
            "kernel_init" : "random", # zeros, random
            "conv_blocks"    : ([["Skip"],["Conv", 32, 3, 1, 'relu'],["Conv", 32, 3, 1, 'relu'],["MaxP", 2],["Skip"],],
                                [["Conv", 32, 3, 1, 'relu'],["Conv", 32, 3, 1, 'relu'],["MaxP", 2],["Skip"],],),

        },


    }

    # Add leave-one-out splits order: MIROC6, CanESM, MPI, MIROC-ES2L, CESM
    for main_key in ['main',]:
        for ileave in range(5):
            leavekey = main_key + '_leave' + str(ileave)
            train_list = list(range(5))
            train_list.remove(ileave)
            split_models = (train_list, train_list, [ileave])
            # Add seeds
            for seed in range(10):
                seedkey = leavekey + '_seed' + str(seed)
                settings_dict[seedkey] = settings_dict[main_key].copy()
                settings_dict[seedkey]['seed'] = seed
                settings_dict[seedkey]['split_models'] = split_models

    # # Make the final one based on main
    for seed in range(10):
        key = 'final_seed' + str(seed)
        settings_dict[key] = settings_dict['main'].copy()
        train_list = list(range(5))
        settings_dict[key]['seed'] = seed
        settings_dict[key]['split_models'] = (train_list, [0,1,2,3,4], [0])

    settings_dict[exp_name]['exp_name'] = exp_name
    return settings_dict[exp_name]