import glob
import numpy as np
import os
import pickle
from multiprocessing import Pool

from pydub import AudioSegment
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import mfcc

base_path = '/media/4tb/data/'
zone = 'Zone1'
write_path = "/home/ayahahmad/pyspeechfeat/"
num_processors = 28
date = "2018_09_01"
extracted_feature = None


def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]: AudioSegment.from_mp3(mp3_file)}


def load_audio(files, num_processors=num_processors):
    files.sort()
    p = Pool(processes=28)
    output = p.map(get_mp3_file, files)
    p.close()
    audio_dict = {k: v for i in output for k, v in i.items()}
    return audio_dict


def pydub_to_np(audio):
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (
                1 << (8 * audio.sample_width)), audio.frame_rate


def sound_to_extracted_feature_array(sound):
    samples, sample_rate = pydub_to_np(sound)
    if extracted_feature == 'mfcc':
        feature_computation = mfcc(samples, sample_rate, nfilt=40, nfft=600)
    elif extracted_feature == 'delta':
        mfcc_feat = mfcc(samples, sample_rate, nfft=600)
        feature_computation = delta(mfcc_feat, 2)
    elif extracted_feature == 'logfbank':
        feature_computation = logfbank(samples, sample_rate, nfft=600)
    return feature_computation


def make_dict_parallel(audio_dict, num_processors=num_processors):
    d_keys = list(audio_dict.keys())
    d_vals = list(audio_dict.values())
    # p = Pool(processes = 8)
    result = map(sound_to_extracted_feature_array, d_vals) # use p.map when running in parallel
    dictionary = {}
    for k, r in zip(d_keys, result):
        dictionary[k] = r
    return dictionary


def date_folder_iter(base_path, zone, daily_dict):
    date_folders = os.listdir(os.path.join(base_path, zone))
    [date_folders.remove(i) for i in date_folders if not i.startswith('2')]
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        files = glob.glob(audio_path + '/2*.mp3')
        audio_dict = load_audio(files)
        daily_dict[folder] = make_dict_parallel(audio_dict)
    if not os.path.exists(f'/{write_path}/{zone}/{extracted_feature}/'):
        os.makedirs(f'/{write_path}/{zone}/{extracted_feature}/')
    with open(f"/{write_path}/{zone}/{extracted_feature}/daily_{extracted_feature}_dict.pkl", "wb") as fil:
        pickle.dump(daily_dict, fil, pickle.HIGHEST_PROTOCOL)
    return daily_dict


if __name__ == "__main__":
    extracted_values = ["delta", "mfcc", "logfbank"]
    for value in extracted_values:
        extracted_feature = f"{value}"
        dictionary = {}
        date_folder_iter(base_path, zone, dictionary)
