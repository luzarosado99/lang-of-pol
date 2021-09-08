s:
# 1. Obtain an exclusive compute node with 28 processors: 
# 
# <code>sinteractive -n 1 -c 28 --partition=broadwl --time=8:00:00 --exclusive</code>
# 
# 2. Load relevant modules:
# 
# <code>module load python</code><br>
# <code>module load ffmpeg</code><br>
# <code>module load praat</code>
# 
# 3. Run below code

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pydub
import parselmouth
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
from multiprocessing import Pool
from collections import defaultdict
from timeit import default_timer as timer
import glob, os, pickle, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import audb
import audiofile
#import opensmile
#import VoiceLab
sns.set()

num_processors = 28
base_path='/project2/graziul/LoP/audio/' # base_path='/project2/mbspencer/nih/data/' ## MODIFIED
#zone='Zone '+str(sys.argv[1])

# Start timer
#start = timer()
zone = 'Zone 1'
date = '2018_08_05'


# In[2]:


# This is extraneous for now
def draw_spectrogram(spect, dynamic_range=70):
    X, Y = spect.x_grid(), spect.y_grid()
    sg_db = 10*np.log10(spect.values)
    min_db = sg_db.max() - dynamic_range
    plt.pcolormesh(X, Y, sg_db, vmin=min_db, cmap='afmhot')
    plt.ylim([spect.ymin, spect.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_pitch(pitch):
    # Extract selected pitch countour, and
    # replace uncoivced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize = 5, color = '2')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize = 2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


# In[2]:


# Create a dictionary {mp3_file:AudioSegment.from_mp3(mp3_file)}
def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]:AudioSegment.from_mp3(mp3_file)}


# In[3]:


# Load audio files
def load_audio(files,num_processors=num_processors):
    files.sort()
    # Create a pool of processors to feed data
    p = Pool(processes = num_processors)
    # Feed the pool a function and objects the function will take as input (this can get tricky)
    output = p.map(get_mp3_file,files)
    # Shut down the pool (always good practice)
    p.close()
    # Reformat output to be more user friendly
    audio_dict = {k:v for i in output for k,v in i.items()}
    return audio_dict


# In[4]:


# Helper function to convert pydub to numpy array (for parselmouth)
def pydub_to_np(audio):
    """Converts pydub audio segment into float32 np array of shape [channels, duration_in_seconds*sampling_rate],
    where each value is in range [-1.0, 1.0]. Returns tuple (audio_np_array, sample_rate)"""
    # get_array_of_samples returns the data in format:
    # [sample_1_channel_1, sample_1_channel_2, sample_2_channel_1, ...]
    # where samples are integers of sample_width bytes.
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (1<<(8*audio.sample_width)), audio.frame_rate


# In[6]:


# This is the code that actually loads the audio data
audio_path = base_path + zone.replace(' ','') + '/' + date + '/'
files = glob.glob(audio_path+'2*.mp3')
audio_dict = load_audio(files)


# In[7]:


# Test code: Can we load an audio file from audio_dict and convert it to numpy format for parselmouth?
sound = audio_dict['201808042331-339616-27730.mp3']
samples, sample_rate = pydub_to_np(sound)


# In[8]:


# Test result: "samples" is a list where each element is a channel - since we have single channel audio data, 
#              have to still pull out that single channel. 
#              "sample_rate" is to help convert between time/frames where each frame is a single sample
samples[0], sample_rate


# In[9]:


# This converts numpy to Praat sound object
snd = parselmouth.Sound(samples[0],  inrt)


# In[13]:


# This extracts amplitude/intensity using Praat
intensity = snd.to_intensity(time_step = snd.sampling_period) 
# Setting time_step = snd.sampling_period gets len(intensity.as_array[0]) to approach len(samples[0])


# In[7]:


# What does the intensity array look like?
intensity.as_array()[0]


# In[15]:


# Converted to series and saved just to see size of file
pd.Series(intensity.as_array()[0]).to_csv(audio_path+'test_intensity.csv',index=False)


# In[6]:


# Suggested to do: 1. Confirm that frame length of "intensity" object is same as frame length of "samples"
#                  2. Create a function to normalize intensity within each file
#                  3. Create a loop to feed audio files to parselmouth and obtain normalized intensity
#                  4. (Optional) Create a matrix or Pandas dataframe with filename as cols and frame intensity as rows
#                  5. Try to parallelize based on structure of "load_audio" function


# ### Extracting Intensity from a Zone

# In[7]:


def sound_to_intensity_array(sound):
    samples, sample_rate = pydub_to_np(sound)
    snd = parselmouth.Sound(samples[0], sample_rate)
    s = time.time()
    intensity = snd.to_intensity() #(time_step = snd.sampling_period)
    intensity_array = intensity.as_array()[0]
    e = time.time()
    print("time to convert intensity into an array: "+str(e-s))
    return intensity_array


# In[5]:


# abstracted 
def sound_to_something_array(sound, to_something):
    samples, sample_rate = pydub_to_np(sound)
    snd = parselmouth.Sound(samples[0], sample_rate)
    something = to_something(snd) #(time_step = snd.sampling_period)
    something_array = something.as_array()[0]
    return something_array


# In[9]:


def peak_normalization(array):
    new_array = (array - array.min())/(array.max() - array.min())
    return new_array


# In[6]:


def make_intensity_single(sound):
    ''' what a single processor will do (the parallel fork)
        input  : sound
        output : normalized_intensity_array
    '''
    temp_intensity_array = sound_to_intensity(sound)
    normalized_intensity_array = peak_normalization(temp_intensity_array)
    return normalized_intensity_array

def make_intensity_dict_parallel(audio_dict, folder, num_processors=num_processors):
    ''' does everything in parallel'''
    
    d_keys = list(audio_dict.keys())
    d_vals = list(audio_dict.values())
    
    s = time.time()
    p = Pool(processes = num_processors)
    result = p.map(make_intensity_single, d_vals)
    e = time.time()
    print("time to make intensities:" +  str(e - s))
    
    dictionary = {}
    s = time.time()
    for k, r in zip(d_keys, result):
        dictionary[k] = r
    p.close()
    e = time.time()
    print("time to make intensity_dict" +  str(e - s))
    
    with open(f"./{folder}_intensity_dict.pkl", "wb") as fil:
        pickle.dump(dictionary, fil, pickle.HIGHEST_PROTOCOL)
    return dictionary
    
def make_intensity_dict_serial(audio_dict):
    ''' does everything in serial'''
    for key in audio_dict.keys():
        sound = audio_dict[key]
        normalized_intensity_array = make_intensity_single(sound)
        intensity_dict[key] = normalized_intensity_array
    return intensity_dict    


# In[24]:


# abstracted
def make_single(sound, to_something=to_something):
    ''' what a single processor will do (the parallel fork)
        input  : sound
        output : normalized_intensity_array
    '''
    temp_array = sound_to_something_array(sound, to_something)
    normalized_array = peak_normalization(temp_array)
    return normalized_intensity_array

def make_dict_parallel(audio_dict, folder, to_something, num_processors=num_processors):
    ''' does everything in parallel'''
    
    d_keys = list(audio_dict.keys())
    d_vals = list(audio_dict.values())
    
    p = Pool(processes = num_processors)
    result = p.map(make_single, d_vals)
    print(result)
    dictionary = {}
    for k, r in zip(d_keys, result):
        dictionary[k] = r
    p.close()
    
    #to_something_name = to_something.replace("parselmouth.Sound.to_", "")
    
    #with open(f"./{folder}_{to_something_name}_dict.pkl", "wb") as fil:
    #    pickle.dump(dictionary, fil, pickle.HIGHEST_PROTOCOL)
    return dictionary


# In[8]:


daily_intensity_dict = {}
def date_folder_iter(base_path, zone):
    date_folders = os.listdir(os.path.join(base_path, zone))
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        files = glob.glob(audio_path+'/2*.mp3')
        start = time.time()
        audio_dict = load_audio(files)
        print("time to create audio_dict:" + str(time.time()-start))
        start = time.time()
        daily_intensity_dict[folder] = make_intensity_dict_parallel(audio_dict, folder)
        print("time to create intensity_dict:" + str(time.time()-start))
    s = time.time()
    with open(f"./{zone}_daily_intensity_dict.pkl", "wb") as fil:
        pickle.dump(daily_intensity_dict, fil, pickle.HIGHEST_PROTOCOL)
    e = time.time()
    print("time to write pkl file:" + str(e - s))
    return daily_intensity_dict


# In[29]:


# abstracted
daily_intensity_dict = {}
daily_harmonicity_dict = {}
daily_pitch_dict = {}
to_something = parselmouth.Sound.to_pitch
def date_folder_iter(base_path, zone, daily_dict, to_something):
    date_folders = os.listdir(os.path.join(base_path, zone))
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        files = glob.glob(audio_path+'/2*.mp3')
        start = time.time()
        audio_dict = load_audio(files)
        print("time to create audio_dict:" + str(time.time()-start))
        start = time.time()
        daily_dict[folder] = make_dict_parallel(audio_dict, folder, to_something)
        print("time to create dict parallel:" + str(time.time()-start))
    s = time.time()
    to_something_name = to_something.replace("parselmouth.Sound.to_", "")
    with open(f"./{zone}_daily_{to_something_name}_dict.pkl", "wb") as fil:
        pickle.dump(daily_dict, fil, pickle.HIGHEST_PROTOCOL)
    e = time.time()
    print("time to write pkl file:" + str(e - s))
    return daily_dict


# In[14]:


import time
s1 = time.time()
zone = "Zone1"
if __name__ == "__main__":
    date_folder_iter(base_path, zone, daily_harmonicity_dict, parselmouth.Sound.to_harmonicity)
e1 = time.time()
print("to run everything: " + str(e-s))


# ### Extracting Harmonicity

# In[26]:


import time
zone = "Zone1"
s1 = time.time()
if __name__ == "__main__":
    date_folder_iter(base_path, zone, daily_harmonicity_dict, parselmouth.Sound.to_harmonicity)
e1 = time.time()
print("to run everything: " + str(e1-s1))


# ### Extracting Pitch

# In[ ]:


zone = "Zone1"
if __name__ == "__main__":
    date_folder_iter(base_path, zone, daily_pitch_dict, parselmouth.Sound.to_pitch)
e1 = time.time()


# ### Extracting Shimmer

# In[ ]:





# ### Ignore below here for now

# In[ ]:


# This extract pitch information
pitch = snd.to_pitch_ac(time_step=0.005, max_number_of_candidates=5000)


# In[10]:


len(pitch)


# In[11]:


len(pitch.to_array())


# In[12]:


pickle.dump(pitch.to_array(),open(audio_path+'pkl_pitch.pkl','wb'))


# In[16]:


#snd.pre_emphasize()
spectrogram = snd.to_spectrogram(maximum_frequency=8000.0, frequency_step=20.0)
pd.DataFrame(spectrogram.as_array().T)


# In[17]:


df_test = pd.DataFrame(spectrogram.as_array().T)
df_test.to_csv(audio_path+'test_spectrogram.csv',index=False)


# In[18]:


len(snd)/len(spectrogram)


# In[19]:


len(snd)/183117


# In[20]:


df_pitch = pd.DataFrame(columns=['frequency','strength'])
pitch_array = pitch.to_array()
for i in range(len(pitch_array)):
    df_pitch.append(pd.DataFrame(pitch_array[i]))
len(df_pitch)


# In[21]:


pd.DataFrame(pitch_array[0])


# In[22]:


pd.DataFrame(pitch.to_array()[0]).describe()


# In[23]:


pd.DataFrame(pitch.to_array()[1]).describe()


# In[24]:


pd.DataFrame(pitch.to_array()[2]).describe()


# In[ ]:





# In[ ]:





# In[25]:


pitch.to_array()


# In[26]:


pitch.to_matrix()


# In[27]:


plt.figure()
draw_spectrogram(spectrogram)


# In[28]:


spectrogram.get_frame_number_from_time(1)
spectrogram.to_spectrum_slice(498)


# In[29]:


plt.figure()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show()


# In[30]:


plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show()


# In[ ]:

