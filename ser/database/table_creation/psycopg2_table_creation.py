import functools
import os
import pickle
import pprint
import psycopg2


def invocation_counter(func):  # to check the number of invocations with the number of items in the SQL table
    inv_counter = 0

    @functools.wraps(func)
    def decorating_function(*args, **kwargs):
        nonlocal inv_counter
        inv_counter += 1
        func(*args, **kwargs)

    def info():
        return inv_counter

    def clear():
        inv_counter = 0

    decorating_function.clear = clear
    decorating_function.info = info
    return decorating_function


### customization of tables ###
table1_name = "individual_audiofile_metadata"
table2_name = "vad_metadata"
table3_name = "daily_audio_metadata"

### filepaths/locations ###
core_path = '/media/4tb/data/'

### connection to psycopg2 ###
try:  # creates table under "t-9ayah" user -- may block other users from accessing it
    conn = psycopg2.connect(host="lop-db.uchicago.edu", sslrootcert=lop - db.uchicago.edu.ca,
                            sslcert=lop - db.uchicago.edu - cert.pem, sslkey=lop - db.uchicago.edu - key.pem, port=5432,
                            user="t-9ayah", database="lop", dbname="lop")
except:
    conn = psycopg2.connect(
        "host=lop-db.uchicago.edu sslmode=require sslrootcert=lop-db.uchicago.edu.ca sslcert=lop-db.uchicago.edu-cert.pem sslkey=lop-db.uchicago.edu-key.pem port=5432 user=t-9ayah dbname=lop")
curr = conn.cursor()
conn.rollback()
### creation of tables ###
try:
    curr.execute("""
       CREATE TABLE individual_audiofile_metadata(
       file_name varchar(30) PRIMARY KEY,
       file_location varchar(70),
       file_length_seconds integer,
       date timestamp,
       zone integer)
   """)
    conn.commit()
except psycopg2.errors.DuplicateTable:
    pass
except psycopg2.errors.InFailedSqlTransaction as e:
    print("ERROR: ", e)
    conn.rollback()
try:
    curr.execute("""
       CREATE TABLE vad_metadata(
       file_name varchar(30) PRIMARY KEY,
       file_location varchar(70),
       nonsilent_minutes integer,
       nonsilent_slices int[]
       )
   """)
    conn.commit()
except psycopg2.errors.DuplicateTable:
    pass
except psycopg2.errors.InFailedSqlTransaction as e:
    print("ERROR: ", e)
    conn.rollback()
try:
    curr.execute("""
       CREATE TABLE daily_audio_metadata(
       date timestamp,
       directory_location varchar(70) PRIMARY KEY,
       complete_data boolean,
       day_length_minutes integer,
       files_total_silence varchar[],
       has_silent_files boolean)
   """)
    conn.commit()
except psycopg2.errors.DuplicateTable:
    pass
except psycopg2.errors.InFailedSqlTransaction as e:
    print("ERROR: ", e)
    conn.rollback()


### general functions ###
def sql_insertion_input(*args):
    """creates a line of the insert table command"""
    entry = ', '.join([str(i) for i in args])
    return f"{entry}"


def from_pickle_metafile(pickle_path, metadata_endswith):
    """ converts a pickle file to an sql table """
    pkl_fil = open(pickle_path, 'rb')
    dictionary = pickle.load(pkl_fil)
    pkl_fil.close()
    for key in dictionary.keys():
        if key.endswith(".mp3") or key.endswith(".wav"):
            file_location = os.path.join(os.path.split(pickle_path)[0], key)
            prop_dict = dictionary[key]
            if metadata_endswith == "vad_dict.pkl":
                try:
                    vad_insertion_from_properties(key, file_location, prop_dict)
                except Exception as e:
                    print(e)
                    conn.rollback()
                    vad_insertion_from_properties(key, file_location, prop_dict)
            elif metadata_endswith == "metadata_dict.pkl":
                try:
                    insertion_from_properties(key, file_location, zone, prop_dict)
                except psycopg2.errors.UniqueViolation:
                    pass
                except Exception as e:
                    print(e)
                    pass


def zone_folder_iter(base_path, metadata_endswith, isdir=False):
    """ finds all the metadata pickle files in each date directory """
    available_files = os.listdir(base_path)
    available_folders = [i for i in available_files if i.startswith("Zone")]
    path_to_folder = [os.path.join(base_path, zone) for zone in available_folders]
    for f in range(len(path_to_folder)):
        global zone
        zone = available_folders[f]
        date_folder_iter(path_to_folder[f], metadata_endswith, isdir)


def date_folder_iter(path_to_folder, metadata_endswith, isdir=False):
    """ finds all the metadata pickle files in each date directory """
    for subdir, directory, files in os.walk(path_to_folder):
        for fil in files:
            if fil.endswith(metadata_endswith) and isdir == False:
                pickle_path = os.path.join(path_to_folder, subdir, fil)
                try:
                    from_pickle_metafile(pickle_path, metadata_endswith)
                except Exception as e:
                    print(e)
                    conn.rollback()
                    from_pickle_metafile(pickle_path, metadata_endswith)
                    pass
            elif fil.endswith(metadata_endswith) and isdir == True:
                pickle_path = os.path.join(path_to_folder, subdir, fil)
                try:
                    daily_from_pickle_metafile(pickle_path, metadata_endswith)
                except psycopg2.errors.InFailedSqlTransaction as e:
                    conn.rollback()
                    daily_from_pickle_metafile(pickle_path, metadata_endswith)


### individual_audiofile_metadata-specific functions ###
@invocation_counter
def insertion_from_properties(file_name, file_location, zone, prop_dict):
    """ uses mp3 properties to generate the insert line """
    recording_start_dict = prop_dict['recording_start']
    day = recording_start_dict['day']
    file_length_seconds = prop_dict.get('file_length_seconds', 0)
    time = recording_start_dict['time']
    year = recording_start_dict['year']
    month = recording_start_dict['month']
    if len(str(time)) == 4:
        date = f"{year}-{month}-{day} {int((str(time))[:2])}:{int((str(time)[2:]))}"
    elif len(str(time)) == 3:
        date = f"{year}-{month}-{day} 0{int((str(time))[:1])}:{int((str(time)[1:]))}"
    elif len(str(time)) == 2:
        date = f"{year}-{month}-{day} 00:{int((str(time)[:2]))}"
    curr.execute(
        f"INSERT INTO {table1_name}(file_name, file_location, file_length_seconds, date, zone) VALUES{file_name, file_location, file_length_seconds, date, int(zone[4:])}")
    conn.commit()


@invocation_counter
def vad_insertion_from_properties(file_name, file_location, vad_prop_dict):
    """ uses mp3 properties to generate the insert line """
    if len(vad_prop_dict) == 0 or file_name.startswith("Zone"):
        return
    vad_program = vad_prop_dict.get('pydub', "other")
    db = vad_program.get(-24, "none")
    nonsilent_minutes = db['nonsilent_minutes']
    nonsilent_slices = db["nonsilent_slices"]
    s = str(nonsilent_slices).replace("[", "{")
    l = s.replace("]", "}")
    slices = str(l)
    curr.execute(
        f"INSERT INTO {table2_name}(file_name, file_location, nonsilent_minutes, nonsilent_slices) VALUES{file_name, file_location, nonsilent_minutes, slices}")
    conn.commit()


### daily_audio_metadata-specific functions ###
@invocation_counter
def daily_insertion_from_properties(directory_location, zone, dictionary, metadata_endswith):
    if len(dictionary) == 0:
        return
    try:
        day_length_minutes = dictionary["day_length_minutes"]
    except Exception as e:
        print(directory_location)
        print(zone)
        pprint.pprint(dictionary)
        return
    try:
        files_total_silence = str(dictionary["files_total_silence"])
    except Exception as e:
        pprint.pprint(dictionary)
        pass
    complete_data = dictionary["complete_data"]
    has_silent_files = dictionary["has_silent_files"]
    directory_name = os.path.split(directory_location)[1]
    f = str(files_total_silence).replace("[", "{")
    i = f.replace("]", "}")
    l = i.replace("'", '"')
    files = str(l)
    date = directory_location.replace("_", "-")
    curr.execute(
        f"INSERT INTO {table3_name} VALUES{str(directory_name), directory_location, complete_data, day_length_minutes, files, has_silent_files}")
    conn.commit()


def daily_from_pickle_metafile(pickle_path, metadata_endswith):
    pkl_fil = open(pickle_path, 'rb')
    dictionary = pickle.load(pkl_fil)
    pkl_fil.close()
    directory_location = os.path.split(pickle_path)[0]
    try:
        daily_insertion_from_properties(directory_location, zone, dictionary,
                                        metadata_endswith)
    except Exception as e:
        print(e)
        conn.rollback()
        daily_insertion_from_properties(directory_location, zone, dictionary,
                                        metadata_endswith)


if __name__ == "__main__":
    zone_folder_iter(core_path, "metadata_dict.pkl")
    print(insertion_from_properties.info())
    zone_folder_iter(core_path, "vad_dict.pkl")
    print(vad_insertion_from_properties.info())
    zone_folder_iter(core_path, "metadata_dict.pkl", True)
    print(daily_insertion_from_properties.info())
