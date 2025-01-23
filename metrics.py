import numpy as np

def R2(Dtrue, Dpred, weighted=False, lats=None, axis=None):
    if weighted:
        assert lats is not None, "must include latitudes for weighting"
        weights = np.cos(np.deg2rad(lats))[None, :, None, None]
    else:
        weights = 1

    eps = np.finfo(float).eps
    ss_res = np.nansum(np.square(Dtrue-Dpred)*weights, axis=axis) # sum of squares of the residual
    ss_tot = np.nansum(np.square(Dtrue-np.nanmean(Dtrue))*weights, axis=axis) # total sum of squares
    return ( 1 - ss_res/(ss_tot + eps) )

def weighted_mean(D, lats, lat_axis=-3, axis=None):
    weights = np.cos(np.deg2rad(lats))
    if lat_axis < 0:
        for i in np.arange(np.abs(lat_axis)-1):
            weights = weights[:, None]
    elif lat_axis >= 0:
        for i in np.arange(len(D.shape) - lat_axis - 1):
            weights = weights[:, None]

    nonanbool = ~np.isnan(D)
    weights = weights * nonanbool
    Dmask = np.ma.masked_array(D, mask=~nonanbool)
    weights = np.ma.masked_array(weights, mask=~nonanbool)
    #D=np.mean(D, axis=axis)
    #weights = np.nanmean(D*weights, axis=axis)
    
    return np.average(Dmask, weights=weights, axis=axis)

def MAE(Dtrue, Dpred, weighted = False, lats=None, axis=None):
    if weighted:
        assert lats is not None, "must include latitudes for weighting"
        weights = np.cos(np.deg2rad(lats))[None, :, None, None]
    else:
        weights = 1
    metrics_arr = np.abs(Dtrue - Dpred)
    nonanbool = ~np.isnan(metrics_arr)
    if weights is not None:
        weights = weights * np.ones_like(nonanbool)
        weights = weights[nonanbool]
    metrics_arr = metrics_arr[nonanbool]
    
    return np.average(metrics_arr, weights=weights, axis=axis)

def globalMAE(Dtrue, Dpred, lats, lat_axis=-3, axis=(3,4,5)):
    return MAE(
                weighted_mean(Dtrue, lats=lats, lat_axis=lat_axis, axis=axis), 
                weighted_mean(Dpred, lats=lats, lat_axis=lat_axis, axis=axis),
              )

def globalE(Dtrue, Dpred, lats, lat_axis=-3, axis=(3,4,5)):
    return np.mean(
                weighted_mean(Dtrue, lats=lats, lat_axis=lat_axis, axis=axis) - \
                    weighted_mean(Dpred, lats=lats, lat_axis=lat_axis, axis=axis),
              )

def MAESS(metric_pred, metric_true, metric_comp, axis=(0,1,2)):
    skill_score = (1 - np.abs(metric_pred - metric_true).mean(axis=axis) / np.abs(metric_comp - metric_true).mean(axis=axis))
    return skill_score