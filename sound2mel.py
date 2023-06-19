import h5py
import numpy as np
import os
import vggish_input


sound_folder = "/Users/xinglidongsheng/ml/kelletal2018-master/demo_stim"

# Get a list of all files in the folder
all_files = os.listdir(sound_folder)

# Filter out only the sound files
sound_files = [os.path.join(sound_folder, f) for f in all_files if f.endswith('.wav')]

audio_input= np.zeros([1,96,64]);
for i, sound_file in enumerate(sound_files):
    # Load your sound file data as a numpy array
    #sound_data, sr = load_sound_file(sound_file)
    audio_one = vggish_input.wavfile_to_examples(sound_file)
    audio_input = np.concatenate((audio_input,audio_one),axis=0)
    #c_gram[1,:,:] = generate_cochleagram(sound_data, sr)
# Convert sound files to .h5 format
# ...
audio_input = np.delete(audio_input,0,axis=0)
# Define the file path and name for the new .h5 file
data_file = os.path.join(sound_folder, 'mel.h5')
# Create a new .h5 file in write mode
with h5py.File(data_file, 'w') as hf:
    # Loop over your sound files and add them as datasets to the .h5 file
    
        # Create a new dataset with a unique name for each sound file
    dataset_name = f'mel'
    hf.create_dataset(dataset_name, data=audio_input)



#with h5py.File(os.path.join(sound_folder, 'sound_files.h5')) as hf:
#    images_n = np.array(hf['sounds']['sounds2'])

