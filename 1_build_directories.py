import os
import directories

dir_settings = directories.get_dirs()

for key in dir_settings:
    if key != "data":
        mkdir_folder = dir_settings[key]
        os.system('mkdir ' + mkdir_folder)