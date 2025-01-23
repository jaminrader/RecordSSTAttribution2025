import time, os, glob, json
import tensorflow as tf

def save_tf_model(model, model_name, directory, settings):

    # save the tf model
    tf.keras.models.save_model(model, directory + model_name + "_model", overwrite=True)

    # save the meta data
    with open(directory + model_name + '_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(settings))

def load_tf_model(model_name, directory):
    # loading a tf model
    model = tf.keras.models.load_model(
        directory + model_name,
        compile=False,
        )
    return model

def load_settings_model(model_name, directory):
    # loading the .json file with settings
    with open(directory + model_name, 'r') as file:
        data = json.load(file)
    return data

def get_model_name(settings):

    model_name = (settings["exp_name"] + '_seed' + str(settings["seed"]))

    return model_name

def save_model(model, settings, base_dirs, start_time=None):

    if start_time == None:
        start_time = time.localtime()
    time_dir = time.strftime("%Y-%m-%d_%H%M", start_time) + '/'

    model_save_loc = base_dirs['models'] + settings['exp_name'] + '/'
    model_save_dir = model_save_loc + time_dir
    os.system('mkdir ' + model_save_loc)
    os.system('mkdir ' + model_save_dir)
    model_name = get_model_name(settings)
    save_tf_model(model, model_name, model_save_dir, settings)

def load_model(EXP_NAME, base_dirs, get_model=True, get_settings=True):
    MODEL_DIRECTORY = base_dirs['models']
    trained_exps = glob.glob(MODEL_DIRECTORY + EXP_NAME + '/*/')
    latest_exp_directory = sorted(trained_exps)[-1] + '/'

    # go through all trained model for the specified experiment
    model_list = []
    settings_list = []
    for name in sorted(os.listdir(latest_exp_directory)):
        if os.path.isdir(latest_exp_directory + name):
            if get_model:
                print(latest_exp_directory + name)
                model_list.append(load_tf_model(name, latest_exp_directory))
        elif name.endswith('metadata.json'):
            if get_settings:
                print(latest_exp_directory + name)
                settings_list.append(load_settings_model(name, latest_exp_directory))
    if get_model and (not get_settings):
        return model_list
    if (not get_model) and get_settings:
        return settings_list
    return model_list, settings_list

