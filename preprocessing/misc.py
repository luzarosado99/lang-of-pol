
import speech_recognition as sr
import pydub
import time
from pydub.silence import split_on_silence, detect_nonsilent
from pydub.playback import play
from pydub import AudioSegment
from pydub.utils import mediainfo
import json
import glob
import pickle
import os
import sys
import numpy as np
import pandas as pd


# Set recognizer function
r = sr.Recognizer()

# Google credentials
GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""{
  "type": "service_account",
  "project_id": "singular-autumn-210720",
  "private_key_id": "447485c5a63248421bcc668cb022e6a2480a31b3",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCftWrjbECuOiux\nG90V4ZtVQ+ywm3Nyo22peOr1DKWI13dFG3ef5qSkJ4y3eMPrEF3t19wl7vsg69XU\nGI8vTKWmb2CEPQTAwVRw8GtwIaxUTT0o69argzdNivIjHs9R69cAvhirbzMY/FWx\nw1k24ZkvPFbw3FSvetvW4N72mt2U+RmnfvdGA4OHmASwPZ5Sy/27ru8IuKDTKWtZ\nR9+lEBWrLWD5yLDB4t/NABaCMR6fOx8UajVeTc+17JPPughBGrBzLDmhmDFEvqya\nUk7BSmb+urDIXNnOX5imE2wweGoeB60Twr9miDEAD3nxvszedoZjgs2o6kySEogG\nv2FWI/6vAgMBAAECggEANDl4Qvb2raJjAC7K3GliSH8GKngix9VuOjFKr6gbh0Ri\nAYyqUPT0WKOALczFUBwRgwGHwTXFE+5ahVkklUR8lJCuIH/tinSNvsK3dzrjpct+\nGOAJ2hr13hr19AsSo0i6DUmcOo8Jx/1XrmHhTgN2eo5CJc/+t0U3FyyAbGaW16wa\nh+V54lJVoPWONA0HBjnNibuX+EEnl+dzxF+DZA0nExUL2i4jshFChnSL/uAXWik2\npIiN+LjYx83LKPbw+ztT+7D2PSi6gn+ugvX4dJhFhLgnMZX2mujstyMqAOCMezAp\naq0JV5ihzNSgCIPjb8Ua/5DZNJS7nJzkFiunH7rzBQKBgQDbrP38aRe/gg2ZmUKB\n28Fl95M1nA2AG3tjsK0ihcFbatnier5eLF84wiKmjOcr8mac4T3p/ajc6kWP0X/4\nFpc8VjJhXKbOo+TE4sZG+nG7LwqcZzBI8IsgOak8CqgU/GON7SMV/cJDVd5LRFdI\nMCpn4iURbjZNz1YrnuqO2fDnkwKBgQC6HftuS68QlVn6HPHphR8O9vcPlXkekQwW\nY72g8R6PuN+nDIAHq7c27K14bCucAcAjQmU0A71abX2KerwIJgSsJM0MtwkgB2hk\nXI7BReiuz0qjgLn7Qu5d8HsQxanBhdtf8JIIGCurk4a4ahFouGWXBbmirCzN+gs5\nLutwCv2F9QKBgB3ZVrWabhCmkkSBr6jHfHLnfgg1yRvUICL+mbfsJsbOMQb2GLHh\nI5spvd2Vnb+58zlz3Z7SycQizQrqs5G5OBmJuNTD6yJ+4JHkIn74fsWpca6o0sXG\ncZESZK104Tvgw4JAa5kMXv4ZR9hAU//KE6kD8Hd620QdXR8WO9bnRDWjAoGAWcqR\nPRsicLu9Vx+Tzne9Djkz5L7WWlrcHAkuuXDar7gfnrY3Jnw/vi3dWxXEzFVD3z7v\nGHMdbX0Zbi/ce4nsAykWDCZaLqukP5cwACq8IWo8tjkqgQA/g+67UNIsHgN4XQbx\nTRpsJzDbdCkoEP+1c3D9qG+shgs2UvZB/CIxQekCgYBsrhWy9fzFAQ4BtSuhmg0M\nwYwLez9U3mcpr/6JYeEj0AboQsrwZnqstsSvXtDb6iHJLIuACK/NFlE+gbAZ7pSL\ndHaaTCqioBlnrfT4xtGyXM0MQf/GO/V3hkQIjqkBRlJIAx5lh8724zyv8ZZakHWE\nHCp+4ZDcJur5FyDLh734/g==\n-----END PRIVATE KEY-----\n",
  "client_email": "lop-83@singular-autumn-210720.iam.gserviceaccount.com",
  "client_id": "117157367570313034840",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/lop-83%40singular-autumn-210720.iam.gserviceaccount.com"
}"""

#Timer function
def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("Transcription took {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# Function for speech to text
def speech_to_text(wav_file, sphinx_only=False, play_audio=False):
	af_wav = sr.AudioFile(wav_file)
	with af_wav as source:
		audio_temp = r.record(source)
	if play_audio:
		play(pydub.AudioSegment.from_wav(wav_file))
	start = time.time()
	temp_dict = {}
	transcription_sphinx = r.recognize_sphinx(audio_temp)
	print("Sphinx: ",transcription_sphinx)
	temp_dict['sphinx'] = transcription_sphinx
	if not sphinx_only:
		# Use Google cloud
		try:
			transcription_google = r.recognize_google_cloud(audio_temp, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
		except:
			transcription_google = ''
		print("Google: ",transcription_google)
		temp_dict['google'] = transcription_google
		# Use IBM
		try:
			transcription_ibm = r.recognize_ibm(audio_temp, username='8d30f0ad-29eb-4184-889b-e8d5ed90a5c0', password='tvmeNWs7uoGo')
		except:
			transcription_ibm = ''
		print("IBM: ",transcription_ibm)
		temp_dict['ibm'] = transcription_ibm
	end = time.time()
	timer(start, end)
	return temp_dict 

##
## Goal: Discrete utterances accurately transcribed with timestamps in a dict-like container
##

#
# Step 1: Record audio and save to file
#

mp3_file = r"\\ssdfiles.uchicago.edu\spencerlab\YMCA\Audio\YMCAStaff1.mp3"
wav_file = mp3_file.replace('.mp3','.wav')
#if not os.path.exists(wav_file):
#	# Read mp3 file
#	mp3 = pydub.AudioSegment.from_mp3(mp3_file)
#	# Convert to wav 
#	mp3.export(wav_file, format='wav')
#wav_file = u"C:/Users/Chris Graziul/Documents/test.wav"

#mp3_audio = pydub.AudioSegment.from_mp3(mp3_file)
#Fs = pydub.utils.mediainfo(mp3_file)['sample_rate']

#wav_audio = pydub.AudioSegment.from_wav(wav_file)

#
# Step 2: Add timestamp
#

# ???

#
# Step 3: Remove silence from file
#

from scipy.io.wavfile import read, write
from scipy import signal
import numpy as np

#wav_file = 'test2.wav'
# Read file
Fs, y = read(wav_file)
# Delete if amplitude = 0
#trim_wav = np.int16(y[abs(y)>0])
# Could also do this in future: floor = mode(abs(v[abs(v)>0]))

#
# Step 4: Use chirp to segment file into chunks
#

# Two types of chirp for separating chunks

# Type A: 	Dispatch-side (high amplitude pulse, followed by beep)
# 			Duration: 0.03s
#			Window:	661.5 ~ 662  

y_find_peaks = signal.find_peaks_cwt(y,[662])

# Type B: Field-side (lower amplitude modulated pulse)

from numpy.lib.stride_tricks import as_strided

def polyphase_core(x, m, f):
    #x = input data
    #m = decimation rate
    #f = filter
    #Force it to be 1D
    x = x.ravel()
    #Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    polyphase = p = np.zeros((m, int((x.shape[0] + f.shape[0]) / m)), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    #Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p

def wavelet_lp(data, ntaps=4):
    #type == 'haar':
    f = np.array([1.] * ntaps)
    return np.sum(polyphase_core(data, 2, f), axis=0)

def wavelet_hp(data, ntaps=4):
    #type == 'haar':
    if ntaps % 2 is not 0:
        raise ValueError("ntaps should be even")
    half = ntaps // 2
    f = np.array(([-1.] * half) + ([1.] * half))
    return np.sum(polyphase_core(data, 2, f), axis=0)

def wavelet_filterbank(n, data):
    #Create and store all coefficients to level n
    x = data
    all_lp = []
    all_hp = []
    for i in range(n):
        c = wavelet_lp(x)
        x = wavelet_hp(x)
        all_lp.append(c)
        all_hp.append(x)
    return all_lp, all_hp

def zero_crossing(x):
    x = x.ravel()
    #Create an X, 2 array of overlapping points i.e.
    #[1, 2, 3, 4, 5] becomes
    #[[1, 2],
    #[2, 3],
    #[3, 4],
    #[4, 5]]
    o = as_strided(x, shape=(x.shape[0] - 1, 2), strides=(x.itemsize, x.itemsize))
    #Look for sign changes where sign goes from positive to negative - this is local maxima!
    #Negative to positive is local minima
    return np.where((np.sum(np.sign(o), axis=1) == 0) & (np.sign(o)[:, 0] == 1.))[0]

def peak_search(hp_arr, arr_max):
    #Given all hp coefficients and a limiting value, find and return all peak indices
    zero_crossings = []
    for n, _ in enumerate(hp_arr):
        #2 ** (n + 1) is required to rescale due to decimation by 2 at each level
        #Also remove a bunch of redundant readings due to clip using np.unique
        zero_crossings.append(np.unique(np.clip(2 ** (n + 1) * zero_crossing(hp_arr[n]), 0, arr_max)))

    #Find refined estimate for each peak
    peak_idx = []
    for v in zero_crossings[-1]:
        v_itr = v
        for n in range(len(zero_crossings) - 2, 0, -1):
            v_itr = find_nearest(v_itr, zero_crossings[n])
        peak_idx.append(v_itr)
    #Only return unique answers
    return np.unique(np.array(peak_idx, dtype='int32'))
    
def find_nearest(v, x):
    return x[np.argmin(np.abs(x - v))]
    
def peak_detect(data, depth):
    if depth == 1:
        raise ValueError("depth should be > 1")
    #Return indices where peaks were found
    lp, hp = wavelet_filterbank(depth, data)
    return peak_search(hp, data.shape[0] - 1)

indices = peak_detect(y,2)

# Questions: 
#	Do all field transmissions have a Type B at beginning and end?
#	Do field transmissions have a beep?



# Split audio on extreme amplitude, short duration signals (dispatcher)

	# Look for outlier amplitudes (define or set?)
	# Try 4-5 std dev 
	# Needs to be a window (need to remove all of the chirp)

outlier = 4*np.std(trim_wav)
df = pd.DataFrame(trim_wav, columns=['f1'])
df_outlier = df[df['f1'].abs()>outlier]


trim_wav = 

# Save file
trimmed_file = wav_file.replace('.wav','-trim.wav')
write(trimmed_file, Fs, trim_wav)


# Find baseline noise profile for each chunk

	# FFT?
	# Audacity noise profile identification, then removal (reverse engineer)

# Remove noise from each chunk



#
# Step 5: Transcribe utterances
#


#wav_file = "C:\\Users\\Chris Graziul\\SDRTrunk\\recordings\\System_Site_District 19__TO_No Squelch_20180731_113426.555.wav"
wav_file = "C:/Users/Chris Graziul/Documents/LoP/test.wav"




'''
#
# Use silence (automatic detection) to break up wav file
#

# Convert to wav (small files)
#sos_mp3 = split_on_silence(mp3)

#for idx, clip in enumerate(sos_mp3):
#	clip.export(wav_file.replace('.wav','-%s.wav' % (str(idx))), format='wav')
#list_of_files = glob.glob(mp3_file.replace('.mp3','*.wav'))
'''

wav_file = r"\\ssdfiles.uchicago.edu\spencerlab\YMCA\Audio\YMCAStaff1nonoise.wav"

wav_file = "/media/sf_Audio/YMCAStaff1nonoise.wav"
seg_text = speech_to_text(wav_file)

# Process each segment
#num_segments = len(sos_mp3)
seg_text = {}

#for seg_num in range(num_segments):
for seg_num in range(len(list_of_files)-1):
	seg_file = wav_file.replace('.wav','-%s.wav' % (str(seg_num)))
	print("Segment %s:" % (str(seg_num)))
	seg_text[seg_file] = speech_to_text(seg_file)
	time.sleep(1)
#pickle.dump(seg_text, open('/home/chris/Downloads/seg_text_dict.pickle','wb'))
