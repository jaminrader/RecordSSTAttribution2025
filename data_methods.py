import numpy as np
import xarray as xr
import metrics

def reshape_from_forcesmip(tr, va):
    # Combine validation and training sets from ForceSMIP preprocessed data
    # final shape is (# climate models, # members, years, lat, lon, variables)
    tr_newshape = (5, 18, tr.shape[0]//5//18) + tr.shape[1:]
    va_newshape = (5, 7, va.shape[0]//5//7) + va.shape[1:]
    tr = tr.reshape(tr_newshape)
    va = va.reshape(va_newshape)
    return np.concatenate([tr, va], axis=1) 

def get_preprocessed_data(filepath):
    npzdat = np.load(filepath)

    if 'oneFpattern' in filepath: # single-pattern data for evaluation of method
        A, F, I, = (npzdat['A'], npzdat['F'], npzdat['I'],)
    else:
        Atr, Ftr, Itr, Ava, Fva, Iva, = (npzdat['Atr'], npzdat['Ftr'], npzdat['Itr'],
                                        npzdat['Ava'], npzdat['Fva'], npzdat['Iva'],)
        A = reshape_from_forcesmip(Atr, Ava)
        F = reshape_from_forcesmip(Ftr, Fva)
        I = reshape_from_forcesmip(Itr, Iva)
    lats = np.linspace(-90, 90, 73)
    lats = [(ll+lh)/2 for ll, lh in zip(lats[:-1], lats[1:])]
    lons = np.linspace(0,360, 145)[:-1]

    if 'ext' in filepath:
        yrs = np.arange(1880,2101)
        yrbool = np.logical_and(yrs>=1950, yrs<=2030)
        A = A[:,:,yrbool,...]
        F = F[:,:,yrbool,...]
        I = I[:,:,yrbool,...]

    return A, F, I, lats, lons

def get_obs(filepath):
    ds = xr.open_dataset(filepath)
    O = ds.sst.values[None, None, ..., None] - 273.15 # model (1), member (1), year (many), lat, lon, 
    if 'ersstv5' in filepath:
        O = O + 273.15
    if 'monthly' in filepath:
        O = np.moveaxis(O.reshape((1,1,-1,12,) + O.shape[-3:]), 3, -1)
        O = O[...,0,:] # get rid of second-to-last dim
        if 'quarterly' in filepath:
            O = O.reshape(O.shape[:-1] + (4,3,)).mean(axis=(-1))
        O = np.concatenate((np.mean(O, axis=-1)[..., None], O), axis=-1)
    return O

def split_component(C, settings):
    # Split a component into its train/val/test splits
    Ctr = C[settings['split_models'][0]][:, settings['split_members'][0]]
    Cva = C[settings['split_models'][1]][:, settings['split_members'][1]]
    Cte = C[settings['split_models'][2]][:, settings['split_members'][2]]

    return Ctr, Cva, Cte

def split_data(A, F, I, settings):
    # Split the data into train/val/test splits
    Atr, Ava, Ate = split_component(A, settings)
    Ftr, Fva, Fte = split_component(F, settings)
    Itr, Iva, Ite = split_component(I, settings)

    return Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite

# Methods for Data Augmentation

def bootstrap(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite):
    # Combine I and F from different climate models for the training set
    Ftr_roll = Ftr.copy()
    for i in np.arange(Ftr.shape[0]-1):
        Ftr_roll = np.roll(Ftr_roll, 1, axis=0)
        Ftr=np.concatenate([Ftr, Ftr_roll], axis=0)
    Itr = np.concatenate([Itr]*Itr.shape[0], axis=0)
    Atr = Itr + Ftr
    return Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite

def augment_data(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite, settings):
    if settings['data_aug_method'] == None:
        pass
    elif settings['data_aug_method'] == 'Itrain2Itest':
        Fte = Fte[:,0:1, ...] + Itr - Itr
        Ite = Itr
        Ate = Fte + Ite
    elif settings['data_aug_method'] == 'Ftrain2Ftest':
        min_mems = np.min([Ftr.shape[1], Ite.shape[1]])
        Fte = Ftr[:, :min_mems]
        Ite = Ite[:, :min_mems] + Ftr[:, :min_mems] - Ftr[:, :min_mems]
        Ate = Fte[:, :min_mems] + Ite[:, :min_mems]
    elif settings['data_aug_method'] == 'bootstrap':
        Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite = bootstrap(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite)
    elif settings['data_aug_method'] == 'bootstrap_Itrain2Itest':
        Fte = Fte[:,0:1, ...] + Itr - Itr
        Ite = Itr
        Ate = Fte + Ite
        Atr, Ava, __, Ftr, Fva, __, Itr, Iva, __ = bootstrap(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite)
    elif settings['data_aug_method'] == 'bootstrap_Ftrain2Ftest':
        min_mems = np.min([Ftr.shape[1], Ite.shape[1]])
        Fte = Ftr[:, :min_mems]
        Ite = Ite[:, :min_mems] + Ftr[:, :min_mems] - Ftr[:, :min_mems]
        Ate = Fte[:, :min_mems] + Ite[:, :min_mems]
        Atr, Ava, __, Ftr, Fva, __, Itr, Iva, __ = bootstrap(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite)

    
    return Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite


# Methods for altering the forced response

def alter_forced(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite, settings):
    if settings['alter_forced_method'] == None:
        pass
    elif settings['alter_forced_method'] == 'shuffle_within_year':
        Ftrshuf = Ftr.flatten()
        np.random.shuffle(Ftrshuf)
        Ftr = Ftrshuf.reshape(Ftr.shape)

        Atr = Itr + Ftr
    elif settings['alter_forced_method'] == 'add_shuffle_between_gridpoints':
        Ftrshuf = Ftr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        Ftrshuf = at_gridpoint_shuffle(Ftrshuf)

        Ftr = np.concatenate([Ftr, Ftrshuf], axis=0)
        Itr = np.concatenate([Itr, Itr], axis=0)
        Atr = Itr + Ftr
    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints':
        Ftrshuf = Ftr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        Ftrshuf = at_gridpoint_shuffle(Ftrshuf)

        Ftr = Ftrshuf
        Atr = Itr + Ftr

    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints_each_model':
        Ftrshuf = Ftr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = Ftrshuf
        Atr = Itr + Ftr

    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints_each_model_val':
        Fvashuf = Fva.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Fvashufi in enumerate(Fvashuf):
            Fvashuf[i] = at_gridpoint_shuffle(Fvashufi[None, ...])

        Fva = Fvashuf
        Ava = Iva + Fva

    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints_each_model_val_internal':
        Ivashuf = Iva.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ivashufi in enumerate(Ivashuf):
            Ivashuf[i] = at_gridpoint_shuffle(Ivashufi[None, ...])

        Iva = Ivashuf
        Ava = Iva + Fva #CHANGED A[..., 0:1] = Iva + Fva

    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints_each_model_train_internal':
        Itrshuf = Itr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Itrshufi in enumerate(Itrshuf):
            Itrshuf[i] = at_gridpoint_shuffle(Itrshufi[None, ...])

        Itr = Itrshuf
        Atr = Itr + Ftr

    elif settings['alter_forced_method'] == 'add_shuffle_between_gridpoints_each_model':
        Ftrshuf = Ftr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = np.concatenate([Ftr, Ftrshuf], axis=0)
        Itr = np.concatenate([Itr, Itr], axis=0)
        Atr = Itr + Ftr


    elif settings['alter_forced_method'] == 'no_forced':
        Ftr = np.broadcast_to(np.nanmean(Ftr, axis=(2))[:,:,None,...], Ftr.shape)
        Fva = np.broadcast_to(np.nanmean(Fva, axis=(2))[:,:,None,...], Fva.shape)
        Fte = np.broadcast_to(np.nanmean(Fte, axis=(2))[:,:,None,...], Fte.shape)

        Atr = Ftr + Itr
        Ava = Fva + Iva
        Ate = Fte + Ite

    elif settings['alter_forced_method'] == 'no_train_forced':
        Ftr = np.broadcast_to(np.nanmean(Ftr, axis=(2))[:,:,None,...], Ftr.shape)
        Atr = Ftr + Itr

    elif settings['alter_forced_method'] == 'add_noise_time':
        noise = np.random.standard_normal(Ftr.shape)[:,:,0:1,...]
        Ftr = Ftr + noise
        Atr = Ftr + Itr
    elif settings['alter_forced_method'] == 'add_training_noise':
        noise = np.random.standard_normal(Ftr.shape)
        Ftr = Ftr + noise
        Atr = Ftr + Itr
    elif settings['alter_forced_method'] == 'add_random_constant_to_map':
        random_constant = np.broadcast_to(np.random.standard_normal(Ftr.shape[0:3])[..., None, None, None], Ftr.shape)
        Ftr = Ftr + random_constant
        Atr = Ftr + Itr
    elif settings['alter_forced_method'] == 'add_random_constant_to_map_trainval':
        random_constant = np.broadcast_to(np.random.standard_normal(Ftr.shape[0:3])[..., None, None, None], Ftr.shape)
        Ftr = Ftr + random_constant
        Atr = Ftr + Itr
        random_constant = np.broadcast_to(np.random.standard_normal(Fva.shape[0:3])[..., None, None, None], Fva.shape)
        Fva = Fva + random_constant
        Ava = Fva + Iva
    elif settings['alter_forced_method'] == 'add_shuffle_between_gridpoints_each_model_and_random_constant_to_map':
        
        Ftrshuf = Ftr.copy()

        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = np.concatenate([Ftr, Ftrshuf], axis=0)
        Itr = np.concatenate([Itr, Itr], axis=0)
        Atr = Itr + Ftr

        random_constant = np.broadcast_to(np.random.standard_normal(Ftr.shape[0:3])[..., None, None, None], Ftr.shape)
        Ftr = Ftr + random_constant
        Atr = Ftr + Itr

    elif settings['alter_forced_method'] == 'shuffle_between_gridpoints_each_model_and_random_constant_to_map':
        
        Ftrshuf = Ftr.copy()

        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = Ftrshuf
        Atr = Itr + Ftr
        
        random_constant = np.broadcast_to(np.random.standard_normal(Ftr.shape[0:3])[..., None, None, None], Ftr.shape)
        Ftr = Ftr + random_constant
        Atr = Ftr + Itr

    elif settings['alter_forced_method'] == 'constant_f_trainval':
        random_constant = np.broadcast_to(np.random.standard_normal(Ftr.shape[0:3])[..., None, None, None], Ftr.shape)
        Ftr = random_constant
        Atr = Ftr + Itr

        random_constant = np.broadcast_to(np.random.standard_normal(Fva.shape[0:3])[..., None, None, None], Fva.shape)
        Fva = random_constant
        Ava = Fva + Iva

    elif settings['alter_forced_method'] == 'concat_shuffle_between_gridpoints_each_model':
        Ftrshuf = Ftr.copy()
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = np.concatenate([Ftr, Ftrshuf], axis=-1)
        Atr = Itr + Ftr     
        Ftr = Ftr[..., :1]

    elif settings['alter_forced_method'] == 'concat_shuffle_between_gridpoints_each_model_trainvaltest':
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        
        Ftrshuf = Ftr.copy()
        for i, Ftrshufi in enumerate(Ftrshuf):
            Ftrshuf[i] = at_gridpoint_shuffle(Ftrshufi[None, ...])

        Ftr = np.concatenate([Ftr, Ftrshuf], axis=-1)
        Atr = Itr + Ftr     
        Ftr = Ftr[..., :1]

        Fvashuf = Fva.copy()
        for i, Fvashufi in enumerate(Fvashuf):
            Fvashuf[i] = at_gridpoint_shuffle(Fvashufi[None, ...])

        Fva = np.concatenate([Fva, Fvashuf], axis=-1)
        Ava = Iva + Fva     
        Fva = Fva[..., :1]

        Fteshuf = Fte.copy()
        for i, Fteshufi in enumerate(Fteshuf):
            Fteshuf[i] = at_gridpoint_shuffle(Fteshufi[None, ...])

        Fte = np.concatenate([Fte, Fteshuf], axis=-1)
        Ate = Ite + Fte     
        Fte = Fte[..., :1]

    elif settings['alter_forced_method'] == 'addconcat_shuffle_between_gridpoints_trainvaltest':
        def at_gridpoint_shuffle(a):
            a_shape = a.shape
            a = a.reshape((-1, a_shape[3]* a_shape[4] * a_shape[5]))
            idx = np.random.rand(*a.shape).argsort(0)
            out = a[idx, np.arange(a.shape[1])]
            out = out.reshape(a_shape)
            return out
        
        Ftrshuf = Ftr.copy() - Ftr[:,:,:40].mean(axis=2)[:,:,None,...]
        Ftrshuf = at_gridpoint_shuffle(Ftrshuf)

        # Add and concatenate Ftrshuf
        Ftr = np.concatenate([Ftr, Ftr + Ftrshuf], axis=-1)
        Atr = Itr + Ftr     
        Ftr = Ftr[..., :1]

        Ftrshuf_allsamples = Ftrshuf.reshape((-1, Ftrshuf.shape[3], Ftrshuf.shape[4], Ftrshuf.shape[5]))

        #Add and concatenate Ftrshuf to Fva and Fte
        Fvashuf_allsamples = Ftrshuf_allsamples[np.random.choice(Ftrshuf_allsamples.shape[0],size=(np.product(Fva.shape[0:3])))]
        Fvashuf = Fvashuf_allsamples.reshape(Fva.shape)

        Fteshuf_allsamples = Ftrshuf_allsamples[np.random.choice(Ftrshuf_allsamples.shape[0],size=(np.product(Fte.shape[0:3])))]
        Fteshuf = Fteshuf_allsamples.reshape(Fte.shape)

        Fva = np.concatenate([Fva, Fva + Fvashuf], axis=-1)
        Ava = Iva + Fva     
        Fva = Fva[..., :1]

        Fte = np.concatenate([Fte, Fte + Fteshuf], axis=-1)
        Ate = Ite + Fte     
        Fte = Fte[..., :1]

    elif settings['alter_forced_method'] == 'concat_n2s_ratio':
        noisemap = np.mean(np.std(Itr[:, :, :, :, :, :], axis=(1,2)), axis=(0))
        signalmap = np.mean(np.std(Ftr[:, :, :, :, :, :], axis=(1,2)), axis=(0))
        n2smap = noisemap/signalmap

        Atr = np.concatenate([Atr, Atr * n2smap], axis=-1)
        Ava = np.concatenate([Ava, Ava * n2smap], axis=-1)
        Ate = np.concatenate([Ate, Ate * n2smap], axis=-1)

    #     Ftr = np.concatenate([Ftr, Atr], axis=-1)
    #     Atr = Itr + Ftr   
    #     Ftr = Ftr[..., :1]

    #     Fva= np.concatenate([Fva, Fva * n2smap], axis=-1)
    #     Ava = Iva + Fva     
    #     Fva = Fva[..., :1]

    #     Fte = np.concatenate([Fte, Fte * n2smap], axis=-1)
    #     Ate = Ite + Fte     
    #     Fte = Fte[..., :1]

    return Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite

# Methods for altering the input maps

def alter_input(Xtr, Xva, Xte, Xob, settings):
    if settings['alter_input_method'] == 'add_random_constant_to_map':
        random_constant = np.broadcast_to(np.random.standard_normal(Xtr.shape[0])[:, None, None, None], Xtr.shape)
        Xtr = Xtr + random_constant
    elif settings['alter_input_method'] == 'double_up':    
        Xtr_orig = Xtr.copy()
        Xtr = np.broadcast_to(Xtr, Xtr.shape[:-1] + (2,))
        Xva = np.broadcast_to(Xva, Xva.shape[:-1] + (2,))
        Xte = np.broadcast_to(Xte, Xte.shape[:-1] + (2,))
        Xob = np.broadcast_to(Xob, Xob.shape[:-1] + (2,))

        print('Should be True:', np.all(Xtr == Xtr_orig))
    elif settings['alter_input_method'] == 'add_random_all':
        random_values = np.random.standard_normal(Xtr.shape) * 1
        Xtr = np.concatenate([Xtr, Xtr + random_values], axis=-1)
        random_values = np.random.standard_normal(Xva.shape) * 1
        Xva = np.concatenate([Xva, Xva + random_values], axis=-1)
        random_values = np.random.standard_normal(Xte.shape) * 1
        Xte = np.concatenate([Xte, Xte + random_values], axis=-1)
        random_values = np.random.standard_normal(Xob.shape) * 1
        Xob = np.concatenate([Xob, Xob + random_values], axis=-1)

    return Xtr, Xva, Xte, Xob


# Methods for standardizing the data (inputs and outputs

def rn2n(D):
    return np.reshape(np.nan_to_num(D), (-1,) + D.shape[-3:])

def get_mask(a, o, settings=None):
    m = np.ones_like(a)
    m[..., np.any(np.isnan(a), axis=(0,1,2), keepdims=False)] = np.nan
    mo = np.ones_like(o)
    mo[..., np.any(np.isnan(o), axis=(0,1,2), keepdims=False)] = np.nan
    m = mo[0:1,0:1,0:1] * m[0:1,0:1,0:1]

    if settings is not None:
        if settings["mask_option"] == None:
            pass
        elif settings["mask_option"] == 'no_poles':
            m[:,:,:,:10,...] = np.nan # get rid of the south pole below 65
            m[:,:,:,-10:, ...] = np.nan # and north above 65
        elif settings["mask_option"] == 'no_highlats':
            m[:,:,:,:15,...] = np.nan # get rid of the south pole below 65
            m[:,:,:,-15:, ...] = np.nan # and north above 65


    return m

normalize = lambda x,mu,sigma: (x-mu)/sigma

def stand(D, basis = None):
    if basis is None:
        Dmean = np.nanmean(D)
        Dstd = np.nanstd(D)
    else:
        Dmean = np.nanmean(basis)
        Dstd = np.nanstd(basis)
    return (D-Dmean)/Dstd, Dmean, Dstd

def gridpoint_stand(D, basis=None):
    if basis is None:
        Dmean = np.nanmean(D, axis=(0,1,2))[None, None, None, ...]
        Dstd = np.nanstd(D, axis=(0,1,2))[None, None, None, ...]
    else:
        Dmean = np.nanmean(basis, axis=(0,1,2))[None, None, None, ...]
        Dstd = np.nanstd(basis, axis=(0,1,2))[None, None, None, ...]
    return (D-Dmean)/Dstd, Dmean, Dstd

class self_standardize:
    def __init__(self, mean_only=False):
        self.mean_only = mean_only
    def __call__(self, D, basis=None, lats=None):
        if lats is None:
            lats = np.linspace(-90, 90, 73)
            lats = [(ll+lh)/2 for ll, lh in zip(lats[:-1], lats[1:])]
        # take global weighted mean across member
        # Dmean = np.nanmean(D, axis=(-3,-2,-1))[..., None, None, None]
        Dmean = metrics.weighted_mean(D, lats, lat_axis=-3, axis=(-3,-2,-1))[..., None, None, None]
        if self.mean_only:
            Dstd = 1
        else:
            Dstd = np.nanstd(D, axis=(-3,-2,-1))[..., None, None, None]
        return (D - Dmean) / Dstd, Dmean, Dstd

class member_standardize:
    def __init__(self, mean_only=False):
        self.mean_only = mean_only
    def __call__(self, D, basis=None):
        if basis is None:
            basis = D
        # take mean across member
        Dmean = np.nanmean(basis, axis=2)[:, :, None, ...]
        if self.mean_only:
            Dstd = 1
        else:
            Dstd = np.nanstd(basis, axis=2)[:, :, None, ...]
        return (D - Dmean)/Dstd, Dmean, Dstd

class stand_self_by_member:
    def __init__(self, mean_only=False,):
        self.mean_only = mean_only
    def __call__(self, D, basis=None):
        # take mean across lat/lon/variable
        Dstand, Dmean1, Dstd1 = self_standardize(mean_only=self.mean_only)(D)
        if basis is not None:
            Dstand_basis, __, __ = self_standardize(mean_only=self.mean_only,)(basis)
        else:
            Dstand_basis = Dstand
        # then across year
        Dstand, Dmean2, Dstd2 = member_standardize(mean_only=self.mean_only)(Dstand, basis=Dstand_basis)
        return Dstand, Dmean1, Dstd1, Dmean2, Dstd2

    # initial shape is model, member, year, lat, lon, variable

class stand_by_member_gridpoint_stand:
    def __init__(self, mean_only=False):
        self.mean_only = mean_only
    def __call__(self, D, basis=None):
        Dstand, Dmean1, Dstd1 = member_standardize(mean_only=self.mean_only)(D, basis=basis)
        if basis is not None:
            Dstand_basis, __, __ = member_standardize(mean_only=self.mean_only,)(basis)
        else:
            Dstand_basis = Dstand
        Dstand, Dmean2, Dstd2 = gridpoint_stand(Dstand, basis=Dstand_basis)
        return Dstand, Dmean1, Dstd1, Dmean2, Dstd2
    
class stand_self_by_member_gridpoint_stand:
    def __init__(self, mean_only=False,):
        self.mean_only = mean_only
    def __call__(self, D, basis=None):
        # take mean across lat/lon/variable
        Dstand, Dmean1, Dstd1 = self_standardize(mean_only=self.mean_only)(D)
        if basis is not None:
            Dstand_basis, __, __ = self_standardize(mean_only=self.mean_only,)(basis)
        else:
            Dstand_basis = Dstand
        # then across year
        Dstand, Dmean2, Dstd2 = member_standardize(mean_only=self.mean_only)(Dstand, basis=Dstand_basis)

        if basis is not None:
            Dstand_basis, __, __ = member_standardize(mean_only=self.mean_only,)(Dstand_basis)
        else:
            Dstand_basis = Dstand
        Dstand, Dmean3, Dstd3 = gridpoint_stand(Dstand, basis=Dstand_basis)
        return Dstand, Dmean1, Dstd1, Dmean2, Dstd2, Dmean3, Dstd3

def no_stand(D):
    Dmean = 0
    Dstd = 1
    return D, Dmean, Dstd

def standardize_inputs(Atr, Ava, Ate, settings, Aob=None, ext_model_idx_overlap=None, obs_model_idx_overlap = [-74, -1],Ftr=None):

    stand_funcs={
        None : no_stand,
        'self_by_member' : stand_self_by_member(),
        'self_by_member_mean_only' : stand_self_by_member(mean_only=True),
        'stand_by_member_gridpoint_stand' : stand_by_member_gridpoint_stand(mean_only=True),
        'stand_self_by_member_gridpoint_stand' : stand_self_by_member_gridpoint_stand(mean_only=True),
        'by_member' : member_standardize(),
        'by_member_mean_only' : member_standardize(mean_only=True),
        'stand' : stand,
        'gridpoint_stand' : gridpoint_stand,
        'self' : self_standardize(),
        'self_mean_only' : self_standardize(mean_only=True)

    }
    # if settings['alter_forced_method'] == 'add_signal_scaled_noise_to_input':
    #     signalmap = np.mean(np.std(Ftr[:, :, :, :, :, :], axis=(1,2)), axis=(0))

    #     Ftrnoise = np.random.standard_normal(Atr.shape) * signalmap
    #     Atr = Atr - Ftrnoise

    #     Fvanoise = np.random.standard_normal(Ava.shape) * signalmap
    #     Ava = Ava - Fvanoise

    #     Ftenoise = np.random.standard_normal(Ate.shape) * signalmap
    #     Ate = Ate - Ftenoise

    # elif settings['alter_forced_method'] == 'concat_signal_noise':
    #     signalmap = np.mean(np.std(Ftr[:, :, :, :, :, :], axis=(1,2)), axis=(0))

    #     Ftrnoise = np.random.standard_normal(Atr.shape) * signalmap
    #     Atr = np.concatenate([Atr, Atr - Ftrnoise], axis=-1)

    #     Fvanoise = np.random.standard_normal(Ava.shape) * signalmap
    #     Ava = np.concatenate([Ava, Ava - Fvanoise], axis=-1)

    #     Ftenoise = np.random.standard_normal(Ate.shape) * signalmap
    #     Ate = np.concatenate([Ate, Ate - Ftenoise], axis=-1)

    #     Fobnoise = np.random.standard_normal(Aob.shape) * signalmap
    #     Aob = np.concatenate([Aob, Aob - Fobnoise], axis=-1)


    func = stand_funcs[settings['in_stand_method']]
    if ext_model_idx_overlap is None:
        Xtr = func(Atr)[0]
        Xva = func(Ava)[0]
        Xte = func(Ate)[0]
    else:
        Xtr = func(Atr, basis=Atr[:,:,ext_model_idx_overlap[0]:ext_model_idx_overlap[1]])[0]
        Xva = func(Ava, basis=Ava[:,:,ext_model_idx_overlap[0]:ext_model_idx_overlap[1]])[0]
        Xte = func(Ate, basis=Ate[:,:,ext_model_idx_overlap[0]:ext_model_idx_overlap[1]])[0]
    Xshapes = Xtr.shape, Xva.shape, Xte.shape

    if Aob is not None:
        Xob = func(Aob, basis=Aob[:,:,obs_model_idx_overlap[0]:obs_model_idx_overlap[1]])[0]
        Xshapes = Xshapes + Xob.shape
        return rn2n(Xtr), rn2n(Xva), rn2n(Xte), rn2n(Xob), Xshapes
    return rn2n(Xtr), rn2n(Xva), rn2n(Xte), Xshapes

def standardize_outputs(Dtr, Dva, Dte, settings,):
    settings_key = 'out_stand_method'
    if settings[settings_key] == None:
        Ttr = Dtr
        Tva = Dva
        Tte = Dte
        Tmean=0
        Tstd=1
    elif settings[settings_key] == 'stand':
        Ttr, Tmean, Tstd = stand(Dtr, basis=Dtr)
        Tva = stand(Dva, basis=Dtr)[0]
        Tte = stand(Dte, basis=Dtr)[0]
    elif settings[settings_key] == 'gridpoint_stand':
        Ttr, Tmean, Tstd = gridpoint_stand(Dtr, basis=Dtr)
        Tva = gridpoint_stand(Dva, basis=Dtr)[0]
        Tte = gridpoint_stand(Dte, basis=Dtr)[0]
    else:
        assert 0 == 1, "Not a valid output standardization method"

    Tshapes = Ttr.shape, Tva.shape, Tte.shape
    return rn2n(Ttr), rn2n(Tva), rn2n(Tte), Tshapes, Tmean, Tstd

def unstandardize_predictions(P, Tmean, Tstd, Tshape, mask):
    return (P.reshape(Tshape) * Tstd + Tmean) * mask

### Function for making a linear fit prediction
def linear_slope(D):
    Mpred = np.zeros_like(D)
    for icm in range(D.shape[0]):
        for imem in range(D.shape[1]):
            for ilat in range(D.shape[3]):
                for ilon in range(D.shape[4]):
                    for ivar in range(D.shape[5]):
                        yridxs = np.arange(D.shape[2])
                        m, b = np.polyfit(yridxs, D[icm, imem, :, ilat, ilon, ivar], deg=1)
                        Mpred[icm, imem, :, ilat, ilon, ivar] = m
    return Mpred

def polyfit_prediction(D, deg=1):
    Lpred = np.zeros_like(D)
    for icm in range(D.shape[0]):
        for imem in range(D.shape[1]):
            for ilat in range(D.shape[3]):
                for ilon in range(D.shape[4]):
                    for ivar in range(D.shape[5]):
                        yridxs = np.arange(D.shape[2])
                        m = np.polyfit(yridxs, D[icm, imem, :, ilat, ilon, ivar], deg=deg)
                        p = np.arange(len(m))[::-1]
                        Lpred[icm, imem, :, ilat, ilon, ivar] = np.sum((yridxs[..., None] ** p[None, ...]) * m[None, ...], axis=1)
    return Lpred

class calc_slopes:
    def __init__(self, slope_len=15):
        self.slope_len = slope_len
    def __call__(self, D):
        slopes_list = []
        for start_yidx in range(D.shape[2]-self.slope_len):
            end_yidx = start_yidx + self.slope_len
            slopes_list.append(linear_slope(D[:,:,start_yidx:end_yidx,...]))
        return slopes_list

def prepare_data(settings, base_dirs,
                 ext_model_idx_overlap = None,
                 obs_model_idx_overlap=[-74, -1]):
    
    A, F, I, lats, lons = get_preprocessed_data(base_dirs['data'] + settings['data_fn'])
    Aob = get_obs(base_dirs['data'] + settings['obs_fn'])

    # Mask the data so they contain like nans
    M = get_mask(A, Aob, settings)
    A = M*A
    F = M*F
    I = M*I
    Aob = M*Aob

    # Make train/test splits
    Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite = split_data(A, F, I, settings)


    # Augment the data (combining F and I from different models)
    Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite = \
    augment_data(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite, settings)

    # Alter the forced response
    Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite = \
    alter_forced(Atr, Ava, Ate, Ftr, Fva, Fte, Itr, Iva, Ite, settings)

    # # CHANGED
    Ftr, Fva, Fte = Ftr[..., 0:1], Fva[..., 0:1], Fte[..., 0:1]
    Itr, Iva, Ite = Itr[..., 0:1], Iva[..., 0:1], Ite[..., 0:1]

    # Standardize the outputs
    Ttr, Tva, Tte, Tshapes, Tmean, Tstd = standardize_outputs(Itr, Iva, Ite, settings,)

    # Standardize the inputs
    Xtr, Xva, Xte, Xob, Xshapes = standardize_inputs(Atr, Ava, Ate, settings, Aob=Aob, Ftr=Ftr,
                                                     ext_model_idx_overlap=ext_model_idx_overlap,
                                                     obs_model_idx_overlap=obs_model_idx_overlap)
    
    # Alter the input maps
    Xtr, Xva, Xte, Xob = alter_input(Xtr, Xva, Xte, Xob, settings)

    return Atr, Ava, Ate, Aob, Ftr, Fva, Fte, Itr, Iva, Ite,\
            Xtr, Xva, Xte, Xob, Xshapes, Ttr, Tva, Tte, Tshapes,\
            Tmean, Tstd, lats, lons, M