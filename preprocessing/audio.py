# To install:
#
# pip install speech_recognition
# apt-get install ffmpeg libav-tools
# pip install pydub 
# apt-get install -qq python python-dev python-pip build-essential swig libpulse-dev libasound2-dev
# pip install pocketsphinx

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
import fnmatch
import shutil


def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush()

##
## Broadcastify archive
## 

# Step 1: Download all files for a certain day for a certain police channel

# Done using scrape.py

# Step 2: Combine files for each day into a single file, removing silence and storing original non-silent ranges 

feeds = {'Zone 1':27730,
	'Zone 2':17684,
	'Zone 3':14545,
	'Zone 4':26296,
	'Zone 5':23006,
	'Zone 6':29284,
	'Zone 8':27158,
	'Zone 10':17693,
	'Zone 11':12444,
	'Zone 12':653,
	'Zone 13':17692}

#This is a modified version of the pydub 'split_on_silence' command that 
#uses the 'detect_nonsilent' command to identify where to split 
def remove_silence(not_silence_ranges, audio_segment, keep_silence):

	audio_nosilence = AudioSegment.empty()
    for start_i, end_i in not_silence_ranges:
        start_i = max(0, start_i - keep_silence)
        end_i += keep_silence
        audio_nosilence+=audio_segment[start_i:end_i]

    return audio_nosilence

#This collects day-level file information and processes audio clips in order 
def combine_remove_silence(feed,date,thresh=-24,save_full=False,delete_incomplete=True,dir_path='C:\\Users\\Chris Graziul\\Documents\\LoP\\Data'):

	feed_spaces = feed
	flushPrint('Removing silence from %s feed, %s' % (feed_spaces, '-'.join(date.split('_'))))
	feed = feed.replace(' ','')
	list_of_files = glob.glob(dir_path+'\\'+feed+'\\'+date+'\\*.mp3')
	list_of_files.sort()
	clips_dict = {mp3_file.split('\\')[-1]:AudioSegment.from_mp3(mp3_file) for mp3_file in list_of_files}

	# Find duration of clips
	length_minutes = round(sum([clip.duration_seconds/60 for clip in list(clips_dict.values())]),1)
	# Note: 60 seconds in a minute, 24*60 = 1440 minutes in a day
	length_days = length_minutes/1440
	if length_days < 0.95:
		print('\nMissing data for %s, only %s of 1440 minutes available' % (date, str(length_minutes)))
		if delete_incomplete:
			for file in list_of_files:
				os.remove(file)			
		return {'length total':length_minutes, 
		'length speech':-1,
		'threshold':thresh, 
		'process time':-1,
		'no silence passages': {}}

	# Combine all 
	if save_full:
		combined = AudioSegment.empty()
		for clip in clips_dict:
		    combined += clips_dict[clip]
		combined.export(dir_path+'\\'+feed+'\\'+date+'\\combo'+date+'.mp3',format='mp3')

	# Combine without silence
	combined_nosilence = AudioSegment.empty()
	combined_detect_nosilence = {}
	start = time.time()
	n=0
	for filename, clip in clips_dict.items():
		n+=1
		flushPrint('Removing silence from %s feed, %s, clip %s of %s' % (feed_spaces, '-'.join(date.split('_')), str(n), str(len(clips_dict))))
		clip_not_silence_ranges = detect_nonsilent(clip, min_silence_len=500, silence_thresh=thresh)
		combined_nosilence += remove_silence(clip_not_silence_ranges, clip, keep_silence=250)
		combined_detect_nosilence[filename] = clip_not_silence_ranges
	combined_nosilence_filename = dir_path+'\\'+feed+'\\'+date+'\\combo'+date+'nosilence'+str(thresh)+'db.mp3'
	if os.path.isfile(combined_nosilence_filename):
		os.remove(combined_nosilence_filename)
	combined_nosilence.export(combined_nosilence_filename,format='mp3')
	end = time.time()
	process_time = round((end-start)/60,1)
	print(' '+str(process_time)+' min')

	length_minutes_nosilence = round(combined_nosilence.duration_seconds/60,1)

	return {'length total':length_minutes, 
		'length no silence':length_minutes_nosilence,
		'threshold':thresh, 
		'process_time':process_time,
		'not_silence_ranges': combined_detect_nosilence}

#This is a wrapper for 'combine_remove_silence' to process all unprocessed days in a feed
def remove_silence_from_feed(feed, thresh=-24, dir_path='C:\\Users\\Chris Graziul\\Documents\\LoP\\Data'):

	#Find days of data
	days = [day.split('\\')[-2] for day in glob.glob(dir_path+'\\'+feed.replace(' ','')+'/*/')]

	#Define filename containing stats and non-silent passages
	feed_day_length_dict_filename = dir_path+'\\'+feed.replace(' ','')+'\\feed_day_length_dict'+str(thresh)+'.pkl'
	#Load dict if file already exists, otherwise initialize dict
	if os.path.isfile(feed_day_length_dict_filename):
		feed_day_length_dict = pickle.load(open(feed_day_length_dict_filename,'rb'))
	else:
		feed_day_length_dict = {}

	#Iterate through days not found in dict	
	for day in days:
		if day in feed_day_length_dict.keys():
			continue
		else:
			feed_day_length_dict[day] = combine_remove_silence(feed=feed, date=day, thresh=thresh, dir_path=dir_path)
			#Save dict as we go
			pickle.dump(feed_day_length_dict, open(feed_day_length_dict_filename,'wb'))

	return print('Completed removing silence from ' + feed + ', processed ' + str(len(feed_day_length_dict)) + ' of data')

remove_silence_from_feed(feed='Zone 3')

# 	Zone 1 - O'Hare, Edison Park, Dunning, Jefferson Park, Portage Park, Old Irving Park, Albany Park
#	Zone 2 - Wrigleyville, Uptown, Lakeview, Ravenswood
# 	Zone 3 - Wicker Park, Bucktown, West Town, Logan Square
#	Zone 4 - Grant Park, Streeterville, River North, Old Town, Lincoln Park
#	Zone 5 - Hyde Park, Bronzeville
#	Zone 6 - Chicago Lawn, Englewood  
#	Zone 7 - NO DATA (Woodlawn, Jackson Park, Southshore)
#	Zone 8 - Auburn/Gresham, Avalon Park, Calumet Heights, South Chicago, Hegewisch
#	Zone 9 - NO DATA (Beverly, Roseland, Morgan Park, Riverdale)
# 	Zone 10 - Lawndale, Garfield Park
#	Zone 11 - Andersonville, Edgewater, Ravenswood, West Ridge
#	Zone 12 - Austin, Belmont, Hermosa
# 	Zone 13 - Back of the Yards, Bridgeport

# Done, 90 days: 2, 3, 4, 5, 6, 8, 10, 11, 13
# Working on, 90 days: 1, 12

#Remove audio files associated with incomplete days
def remove_data_from_incomplete_days(feed, dir_path='C:\\Users\\Chris Graziul\\Documents\\LoP\\Data'):
	feed_spaces = feed
	feed = feed.replace(' ','')
	days = [day.split('\\')[-2] for day in glob.glob(dir_path+'\\'+feed+'/*/')]
	for date in days:
		flushPrint('Removing incomplete days from %s feed, %s' % (feed_spaces, '-'.join(date.split('_'))))
		list_of_files = glob.glob(dir_path+'\\'+feed+'\\'+date+'\\*.mp3')
		if len(list_of_files)>0:
			list_of_files.sort()
			clips_dict = {mp3_file.split('\\')[-1]:AudioSegment.from_mp3(mp3_file) for mp3_file in list_of_files}
			# Find duration of clips
			length_minutes = round(sum([clip.duration_seconds/60 for clip in list(clips_dict.values())]),1)
			# Note: 60 seconds in a minute, 24*60 = 1440 minutes in a day
			length_days = length_minutes/1440
			#Remove days with incomplete data
			if length_days < 0.95:
				for file in list_of_files:
					os.remove(file)

zones = ['Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 8','Zone 10','Zone 11','Zone 13']
for zone in zones:
	remove_data_from_incomplete_days(zone)

#Load dictionary to create dataframe of police chatter
def get_prop_police_talk(list_of_feeds, thresh=-24, dir_path='C:\\Users\\Chris Graziul\\Documents\\LoP\\Data'):
	#Initialize dictionary
	first_dict_file = dir_path+'\\Zone'+str(list_of_feeds[0])+'\\feed_day_length_dict'+str(thresh)+'.pkl'
	first_dict = pickle.load(open(first_dict_file,'rb'))
	df = pd.DataFrame.from_dict(first_dict,orient='index')
	vars_of_interest = ['length no silence','length total']
	df_first = df[vars_of_interest].copy()
	df_first.loc[:,'prop_chatter'] = df_first['length no silence']/df_first['length total']
	df_first.loc[:,'date'] = df_first.index
	df_first.loc[:,'zone'] = 'Zone ' + str(list_of_feeds[0])
	#Loop through remaining feeds (if more than one)
	if len(list_of_feeds) > 1:
		df_multiple = df_first
		remaining_feeds = [i for i in list_of_feeds if i != list_of_feeds[0]]
		for feed in remaining_feeds:
			next_dict_file = dir_path+'\\Zone'+str(feed)+'\\feed_day_length_dict'+str(thresh)+'.pkl'
			next_dict = pickle.load(open(next_dict_file,'rb'))
			df = pd.DataFrame.from_dict(next_dict,orient='index')
			df_next = df[vars_of_interest].copy()
			df_next.loc[:,'prop_chatter'] = df_next['length no silence']/df_next['length total']
			df_next.loc[:,'date'] = df_next.index
			df_next.loc[:,'zone'] = 'Zone ' + str(feed)
			df_multiple = df_multiple.append(df_next)
		return df_multiple.loc[df_multiple['prop_chatter'].notna(), ['date','zone','length total','prop_chatter']].reset_index(drop=True)
	else:
		return df_first[df_first['prop_chatter'].notna(), ['date','zone','length total','prop_chatter']].reset_index(drop=True)

#List of feeds where removal of silence has completed
list_of_feeds=[1, 2, 3, 4, 5, 6, 8, 10, 11, 13]

df = get_prop_police_talk(list_of_feeds)

#Get stats on "police chatter"
def get_chatter_stats(df):
	#Get chatter stats
	chatter_stats = df[['date','zone','prop_chatter']].groupby('zone').describe().copy()['prop_chatter']
	chatter_stats['zone'] = chatter_stats.index
	new_index = chatter_stats['zone'].str.split(' ').apply(lambda x: x[1]).astype(int)
	chatter_stats = chatter_stats[['zone','count','mean','std','min','25%','50%','75%','max']].set_index(new_index).sort_index(axis=0)
	#Get length of non-silence (i.e. speech)
	length_stats = df[['date','zone','length total']].groupby('zone').describe().copy()['length total']
	length_stats['total_minutes'] = round(length_stats['mean']*length_stats['count']/60,0)
	length_stats['zone'] = length_stats.index
	new_index = length_stats['zone'].str.split(' ').apply(lambda x: x[1]).astype(int)
	length_stats = length_stats[['zone','total_minutes']].set_index(new_index).sort_index(axis=0)
	#Combine information
	stats = chatter_stats.merge(length_stats[['total_minutes']],left_index=True, right_index=True)
	stats = stats[['zone','count','total_minutes','mean','std','min','25%','50%','75%','max']]
	#Add totals
	stats=stats.append(df.prop_chatter.describe())
	stats.loc[stats.index=='prop_chatter','zone'] = 'Total'
	stats.loc[stats.index=='prop_chatter','total_minutes'] = length_stats['total_minutes'].sum()
	#Round as needed
	for var in ['mean','std','min','25%','50%','75%','max']:
		stats[var] = stats[var].round(2)
	stats['count'] = stats['count'].astype(int)
	stats['total_minutes'] = stats['total_minutes'].astype(int)
	#Return result
	return stats

stats = get_chatter_stats(df)
stats.to_latex(index=False)

df.to_csv(dir_path+'\\prop_chatter.csv',index=False)

# Offload day data that has been processed to remove silence (to recover storage space for further processing)
def offload_completed_days(feed, drive_letter, dir_path='C:\\Users\\Chris Graziul\\Documents\\LoP\\Data'):
	feed_spaces = feed
	feed = feed.replace(' ','')
	days = [day.split('\\')[-2] for day in glob.glob(dir_path+'\\'+feed+'/*/')]
	for date in days:
		drive_path = "%s:\\Data\\%s\\%s\\" % (drive_letter, feed, date)
		dir_files = [snippet for snippet in glob.glob(dir_path+'\\'+feed+'\\'+date+'/*')]
		is_processed = len([file for file in dir_files if fnmatch.fnmatch(file,'*combo*.mp3')]) > 0
		if is_processed:
			for file in dir_files:
				if fnmatch.fnmatch(file,'*combo*.mp3'):
					print('Moving processed file for '+date.replace('_','-'))
					print(drive_path+file.split('\\')[-1])
					shutil.move(file, drive_path+file.split('\\')[-1])
		#shutil.rmtree(drive_path)	

def fix_my_error(path="E:\\Data\\Zone1"):
	days = [day.split('\\')[-2] for day in glob.glob(path+'/*/')]
	for date in days:
		path_files = glob.glob(path+'\\'+date+'/*')
		for file in path_files:
			if fnmatch.fnmatch(file,'*%s/*%s' % (date, date)):
				print("Found it")
				print(file)
				old_filename = file
				new_filename = '\\'.join(file.split('\\')[:-1])+'\\'+'combo'+date+'nosilence-24db.mp3' 
				os.rename(old_filename,new_filename)
				print("Renamed "+date)

fix_my_error()