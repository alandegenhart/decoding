# Copy experiment data from server
#
# This script copies Energy Landscape data from the Batista server to the standard external hard drive location.
# Standard file lab data naming schemes are used (animal/yyyy/mm/yyyymmdd).

# Import packages
import os
import shutil
import sys

# Define paths and subject/dataset info (hard-coded for now)
subject = sys.argv[1]
dataset = sys.argv[2]
print('Subject: {}'.format(subject))
print('Dataset: {}'.format(dataset))
copy_dir_name = dataset
y = dataset[0:4]
m = dataset[4:6]

# Get all files in the desired directory
local_dir = '/Volumes/Samsung_T5/Batista/Animals/'
server_dir = '/Volumes/Batista/Animals/'
data_dir = os.path.join(server_dir, subject, y, m, dataset)
copy_dir = os.path.join(local_dir, subject, y, m, copy_dir_name)

# Check to make sure that the local and server directories are accessible.  The local directory will not be accessible
# if the SSD is not connected.  The server directory will not be accessible if the server disk isn't mounted beforehand.
if os.path.isdir(server_dir):
    print('Server directory located.')
else:
    print('Server directory not found.  Check that the server disk is mounted.')
    sys.exit(1)

if os.path.isdir(local_dir):
    print('Local directory located.')
else:
    print('Local directory not found.  Check that the external hard drive is connected and mounted.')
    sys.exit(1)

# Check to make sure that the desired dataset exists
if not os.path.isdir(data_dir):
    print('Desired data directory does not exist.')
    sys.exit(1)

# Check to make sure the desired directory to copy to exists.  This is needed when the month/year changes.
mo_dir = os.path.join(local_dir, subject, y, m)
if not os.path.isdir(mo_dir):
    print('Creating directory: {}'.format(mo_dir))
    os.makedirs(mo_dir)  # The 'makedirs()' function will create intermediate directories if needed.

# Check to see if data directory exists and create if not
if not os.path.isdir(copy_dir):
    print('Creating directory: {}'.format(dataset))
    os.mkdir(copy_dir)

# Create translated data directory
trans_dir = os.path.join(copy_dir, 'translated')
if not os.path.isdir(trans_dir):
    print('Creating directory: {}/translated'.format(dataset))
    os.mkdir(trans_dir)

# Copying data in one shot works OK, but means that everything gets copied.
# This is less than ideal for the data directories, which have a bunch of zip
# files that take a long time to copy.  It also means that any Trial and
# TrajectoryData objects get copied, which takes a lot of time.  Instead,
# iterate through and first (1) copy all files.  The iterate through directories
# and copy any translated *.mat files.  The 'analysis' directory also needs to
# be copied
with os.scandir(data_dir) as it:
    for item in it:
        # If the item is a file, copy it
        if item.is_file():
            print('Copying: {}'.format(item.name))
            src_file = os.path.join(data_dir, item.name)
            dest_file = os.path.join(copy_dir, item.name)
            shutil.copy2(src_file, dest_file)

        # If the item is a directory, look for a translated *.mat file
        if item.is_dir():
            # Get list of files in the directory
            dir_path = os.path.join(data_dir, item.name)
            dir_cont = os.listdir(dir_path)

            # Iterate over files in the directory and look for translated *.mat
            # files
            for dir_item in dir_cont:
                if dir_item.endswith('SI_translated.mat'):
                    print('Copying: {}/{}'.format(item.name, dir_item))
                    src_file = os.path.join(dir_path, dir_item)
                    dest_file = os.path.join(trans_dir, dir_item)
                    shutil.copy2(src_file, dest_file)

# Special directories to copy - analysis directory
analysis_dir = os.path.join(data_dir, 'analysis')
if os.path.isdir(analysis_dir):
    print('Copying: analysis')
    dest_dir = os.path.join(copy_dir, 'analysis')
    shutil.copytree(analysis_dir, dest_dir)
