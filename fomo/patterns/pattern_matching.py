import numpy as np
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler, MinMaxScaler, robust_scale, scale, minmax_scale

def stack_time_series(series_list):
    stacked_ser = np.stack(series_list, axis=0)
    return stacked_ser

def find_matching_barycenter(stacked_ser, barycenters, radius=4, override_complete=True, scale_func = None, scale_kwargs={}):
    """Find the best matching barycenter for an array of time series examples
    
    Parameters
    stacked_ser: np.array
        2d array with shape (n time series, len of time series)
    barycenters: np.array
        2d array with shape (n barycenters, len of time series)
    radius: int (default 4)
        sakoe-chibe radius constraint for comparing each time series to the barycenters
    override_complete: bool (default True)
        When true, replaces fully 0 arrays with a unique label, and fully 1 arrays with a unique label
    scale_func: str or function, default None
        Options: 'standard' for standard scale, 'minmax' for minmax scale, or other function
    scale_kwargs: dict
        Dict to pass to scale function
    
    TODO:    
    Add confidence or consistency filter
    
    """
    if scale_func is not None:
        if type(scale_func) == str:
            if scale_func == 'standard':
                stacked_ser = scale(stacked_ser, with_std=True, axis=1, **scale_kwargs)
            elif scale_func == 'minmax':
                stacked_ser = minmax_scale(stacked_ser, axis=1, **scale_kwargs)
        else:
            stacked_ser = scale_func()
            
            
    # Combine into one matrix
    stacked_wpattern = np.concatenate([barycenters, stacked_ser], axis=0)
    
    # Pairwise dtw comparisons, limited (with 'block' argument) to only make necessary comparisons
    dist_lst = dtw.distance_matrix(stacked_wpattern,
                               block=(
                                       (0,len(barycenters)),
                                       (len(barycenters), len(stacked_wpattern))
                                     ), 
                              window=radius,
                              compact=True)
    
    # Reshape so each example in stacked_ser is a column
    dist_mat = np.array(dist_lst).reshape(len(barycenters),-1)
    
    # Find each time series' best match
    best_match = np.argmin(dist_mat, axis=0)
    
    if override_complete:
        best_match_adjusted = best_match.copy()
        time_series_len = stacked_ser.shape[1]
        # adjust full ones
        best_match_adjusted[(stacked_ser == np.ones(time_series_len)).all(axis=1)] = len(barycenters)

        # adjust full zeros
        best_match_adjusted[(stacked_ser == np.zeros(time_series_len)).all(axis=1)] = len(barycenters) + 1

        return best_match_adjusted
    
    return best_match