directory_dict = {
    'data' : 'data/preprocessed/',
    'models': 'saved_models/',
    'figures' : 'figures/',
    'suppfigures' : 'figures/supp/',
    'final_figures' : 'final_figures/',
    'attribution_estimate' : 'attribution_estimate/',
}

def get_dir(key):
    return directory_dict[key]

def get_dirs():
    return directory_dict