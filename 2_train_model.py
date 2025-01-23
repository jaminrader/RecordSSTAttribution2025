import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend
import seaborn as sb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import importlib as imp 
import matplotlib.pyplot as plt
import time
import gc
from save_load import save_model, load_model

import experiments
import directories
import data_methods
import network_methods
import metrics

### To run all leave-one-out experiments
exp_list = []
for main_key in ['main',]:
    for ileave in range(5):
        leavekey = main_key + '_leave' + str(ileave)
        # Add seeds
        for seed in range(10):
            seedkey = leavekey + '_seed' + str(seed)
            exp_list.append(seedkey)
EXPERIMENTS = exp_list

EXPERIMENTS = EXPERIMENTS + ['final_seed0','final_seed1','final_seed2','final_seed3','final_seed4','final_seed5','final_seed6','final_seed7','final_seed8','final_seed9',]

print(EXPERIMENTS)
base_dirs = directories.get_dirs()
start_time = time.localtime()


for EXP in EXPERIMENTS:
    
    print('*** Preparing data for ' + EXP)

    # I.D. experiment settings
    settings = experiments.get_settings(EXP)

    if  settings['data_fn'] == 'standard_tos_annual_ext.npz':
        # ext_model_idx_overlap = [70, 143]
        # obs_model_idx_overlap=[-74, -1]
        ext_model_idx_overlap = [0,73]
        obs_model_idx_overlap=[-74, -1]
    else: 
        ext_model_idx_overlap = None
        obs_model_idx_overlap=[-74, -1]

    Atr, Ava, Ate, Aob, Ftr, Fva, Fte, Itr, Iva, Ite, \
            Xtr, Xva, Xte, Xob, Xshapes, Ttr, Tva, Tte, Tshapes, \
            Tmean, Tstd, lats, lons, M = data_methods.prepare_data(settings, base_dirs,
                                                                   ext_model_idx_overlap = ext_model_idx_overlap,
                                                                   obs_model_idx_overlap = obs_model_idx_overlap)


    # print(Xtr.shape)
    # print(Ttr.shape)

    # import nada

    # Train the model
    print('*** Training model for ' + EXP)
    start_train_time = time.time()
    tf.keras.utils.set_random_seed(settings['seed'])

    model, encoder, decoder = network_methods.train_CVED(Xtr, Ttr, Xva, Tva, settings,verbose=0)
    print('Time elapsed for training: ', time.time() - start_train_time)

    # Make predictions
    print('*** Making predictions for ' + EXP)
    Ptr = model.predict(Xtr)
    _ = gc.collect()
    Pva = model.predict(Xva)
    _ = gc.collect()
    Pte = model.predict(Xte)
    _ = gc.collect()
    PItr = data_methods.unstandardize_predictions(Ptr, Tmean, Tstd, Tshapes[0], M[..., 0:1])
    PIva = data_methods.unstandardize_predictions(Pva, Tmean, Tstd, Tshapes[1], M[..., 0:1])
    PIte = data_methods.unstandardize_predictions(Pte, Tmean, Tstd, Tshapes[2], M[..., 0:1])
    PFtr = Atr[..., 0:1] - PItr
    PFva = Ava[..., 0:1] - PIva
    PFte = Ate[..., 0:1] - PIte

    # Compute Metrics
    results = dict()
    results['wR2_Itr'] = metrics.R2(Itr, PItr, weighted=True, lats=lats)
    results['wR2_Iva'] = metrics.R2(Iva, PIva, weighted=True, lats=lats)
    results['wR2_Ite'] = metrics.R2(Ite, PIte, weighted=True, lats=lats)
    results['wMAE_Itr'] = metrics.MAE(Itr, PItr, weighted=True, lats=lats)
    results['wMAE_Iva'] = metrics.MAE(Iva, PIva, weighted=True, lats=lats)
    results['wMAE_Ite'] = metrics.MAE(Ite, PIte, weighted=True, lats=lats)
    results['gMAE_Itr'] = metrics.globalMAE(Itr, PItr, lats)
    results['gMAE_Iva'] = metrics.globalMAE(Iva, PIva, lats)
    results['gMAE_Ite'] = metrics.globalMAE(Ite, PIte, lats)
    results['gE_Itr'] = metrics.globalE(Itr, PItr, lats)
    results['gE_Iva'] = metrics.globalE(Iva, PIva, lats)
    results['gE_Ite'] = metrics.globalE(Ite, PIte, lats)

    # Save the model
    model_specs = settings.copy()
    model_specs['exp_results'] = results.copy()
    save_model(model, model_specs, base_dirs, start_time=start_time)
