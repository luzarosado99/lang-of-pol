import parselmouth
from pydub import AudioSegment
from multiprocessing import Pool
import glob, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

num_processors = 28
base_path = '/project2/graziul/LoP/audio/'
zone = 'Zone 1'
date = '2018_08_05'
feature = lambda x: x  # placeholder function, to be replaced by feature extraction function later in the code


def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]: AudioSegment.from_mp3(mp3_file)}


def load_audio(files, num_processors=num_processors):
    files.sort()
    p = Pool(processes=num_processors)
    output = p.map(get_mp3_file, files)
    p.close()
    audio_dict = {k: v for i in output for k, v in i.items()}
    return audio_dict


def pydub_to_np(audio):
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (
            1 << (8 * audio.sample_width)), audio.frame_rate


def sound_to_extracted_feature_array(sound, feature):
    samples, sample_rate = pydub_to_np(sound)
    snd = parselmouth.Sound(samples[0], sample_rate)
    extracted_feature = feature(snd)  # (time_step = snd.sampling_period)
    extracted_feature_array = extracted_feature.as_array()[0]
    return extracted_feature_array


def peak_normalization(array):
    new_array = (array - array.min()) / (array.max() - array.min())
    return new_array


def make_single(sound, feature=feature):
    temp_array = sound_to_extracted_feature_array(sound, feature)
    normalized_array = peak_normalization(temp_array)
    return normalized_array


def make_dict_parallel(audio_dict, folder, feature, num_processors=num_processors):
    d_keys = list(audio_dict.keys())
    d_values = list(audio_dict.values())
    p = Pool(processes=num_processors)
    result = p.map(make_single, d_values)
    dictionary = {}
    for k, r in zip(d_keys, result):
        dictionary[k] = r
    p.close()
    f = str(feature)
    feature_name = f.replace("parselmouth.Sound.to_", "")
    
    with open(f"./{folder}_{feature_name}_dict.pkl", "wb") as fil:
        pickle.dump(dictionary, fil, pickle.HIGHEST_PROTOCOL)
    return dictionary


def date_folder_iter(base_path, zone, feature, daily_dict={}):
    date_folders = os.listdir(os.path.join(base_path, zone))
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        files = glob.glob(audio_path + '/2*.mp3')
        audio_dict = load_audio(files)
        daily_dict[folder] = make_dict_parallel(audio_dict, folder, feature)
    to_something_name = feature.replace("parselmouth.Sound.to_", "")
    with open(f"./{zone}_daily_{to_something_name}_dict.pkl", "wb") as fil:
        pickle.dump(daily_dict, fil, pickle.HIGHEST_PROTOCOL)
    return daily_dict


if __name__ == "__main__":
    zone = "Zone1"
    features = [parselmouth.Sound.to_harmonicity, parselmouth.Sound.to_intensity, parselmouth.Sound.to_harmonicity]
    for f in features:
        date_folder_iter(base_path, zone, f)
