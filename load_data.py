# Data conversion script.  This script loads MATLAB *.mat files from disk and converts them to a Python-friendly format.


# Function to load data
def load_trial_data(data_path):

    # Import
    import h5py

    # Load data
    T = h5py.File(data_path, 'r')

    # The HDF5 dataset format is a bit odd coming from MATLAB.  To access top-level data, things are fairly
    # straightforward:
    S = T['S_temp']  # The variable name is a key

    # Get the value associated for the desired keys.  First loop over all keys and add items to a dict if they are a
    # dataset
    ds_type = type(S['subject'])
    trial_data = {}
    for key in S.keys():
        if type(S[key]) == ds_type:
            trial_data[key] = S[key][()]
        else:
            trial_data[key] = None

    # Now go through and convert fields as needed and remove unneeded fields

    # Convert ASCII to strings
    trial_data['subject'] = ''.join([chr(l) for l in trial_data['subject']])
    trial_data['date'] = ''.join([chr(l) for l in trial_data['date']])
    trial_data['trialName'] = ''.join([chr(l) for l in trial_data['trialName']])

    # Convert single-value arrays to single numbers
    trial_data['successful'] = trial_data['successful'][0, 0]
    trial_data['tag'] = trial_data['tag'][0, 0]
    trial_data['targetCode'] = int(trial_data['targetCode'][0, 0])
    trial_data['targetRadius'] = trial_data['targetRadius'][0, 0]
    trial_data['trajectoryOnset'] = trial_data['trajectoryOnset'][0, 0]
    trial_data['trajectoryOffset'] = trial_data['trajectoryOffset'][0, 0]
    trial_data['trialID'] = int(trial_data['trialID'][0, 0])

    # Convert 2D arrays to 1D
    trial_data['stateOnset'] = trial_data['stateOnset'][0, :]
    trial_data['stateOffset'] = trial_data['stateOffset'][0, :]
    trial_data['targetPosition'] = trial_data['targetPosition'][0, :]
    trial_data['time'] = trial_data['time'][0, :]

    # Remove fields
    del trial_data['decodeSpikeCounts']
    del trial_data['decodeState']
    del trial_data['decodeTime']
    del trial_data['decoderName']

    # Two fields need to be handled differently: 'states' (cell array) and 'spikes' (struct array)

    # Get spiking information

    # First, we need to get the HDF5 objects for the elements in the structure:
    spikes = S['spikes']  # A HDF5 group
    chn_ref = spikes['channel'][()][0, :]  # This returns an array of HDF5 object references
    srt_ref = spikes['sort'][()][0, :]
    st_ref = spikes['spikeTimes'][()][0, :]

    # Once we have the objects, we can read the data.  For the channel and sort data we can do this in one shot.
    chn = [T[ref][0, 0] for ref in chn_ref]
    srt = [T[ref][0, 0] for ref in srt_ref]
    st = [T[ref][()] for ref in st_ref]  # In this case we can't yet format the spikes as 1D arrays

    # If the spike times are empty the conversion will result in a 1D array (rather than a 2D array).  Convert 2D arrays
    # to 1D arrays for valid spike times and replace invalid arrays with 'None'
    st = [s[:, 0] if s.ndim == 2 else None for s in st]

    # Add data to dict.  Since this might eventually be converted to a different format there is no need to make things
    # overly complicated by adding another level of abstraction (e.g., a dict)
    trial_data['spike_channel'] = chn
    trial_data['spike_sort'] = srt
    trial_data['spike_times'] = st
    del trial_data['spikes']  # No longer need thi

    # Read state names
    states = trial_data['states'][:, 0]  # Get object references
    states_num = [T[ref][()] for ref in states]  # Get array of ASCII codes for each state name
    trial_data['states'] = [''.join([chr(l) for l in st_ar]) for st_ar in states_num]  # Convert ASCII codes to string

    # Close reference to file
    T.close()

    return trial_data


def main():
    import os
    import pandas as pd

    # Get all valid trial files
    data_path_base = '/Volumes/Samsung_T5/Random Datasets/Ike_PMd_MemoryGuidedReach/mat/'
    all_files = os.listdir(data_path_base)

    # Iterate over trial files and convert
    trial_list = []
    for trial_file in all_files:
        # Load data
        data_path = os.path.join(data_path_base, trial_file)
        trial_data = load_trial_data(data_path)

        # Convert to pandas Dataframe
        trial_list.append(trial_data)

        print('Loading file: {}'.format(trial_file))

    # For each trial file, convert from the dict format to something easier to manipulate (pandas?)
    print('Number of trials load from disk: {}'.format(len(trial_list)))

    # Save data in a python-friendly format
    col = ['subject', 'trialID', 'successful',
           'trajectoryOnset', 'trajectoryOffset', 'stateOnset', 'stateOffset', 'states',  # timing info
           'targetCode', 'targetPosition', 'targetRadius', 'time', 'pos', 'vel',  # kinematic info
           'spike_channel', 'spike_sort', 'spike_times',  # neural info
           'date', 'trialName', 'tag'  # unused info
           ]
    df = pd.DataFrame(trial_list, columns=col)

    # Print out summary of converted data
    print(df)

    # Save data to HDF
    save_path = '/Volumes/Samsung_T5/Random Datasets/Ike_PMd_MemoryGuidedReach/Ike_MGR.hdf'
    df.to_hdf(save_path, 'TrialData')


main()

