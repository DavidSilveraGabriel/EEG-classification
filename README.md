# Using-mne-and-braindecode
in this mini project i will use the libraries [mne](https://mne.tools/dev/index.html) and [braindecode](https://robintibor.github.io/braindecode/index.html) to analyze and classify public domain EEG(ML) signals provided by [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/) here the [paper](https://arxiv.org/pdf/1703.05051.pdf) from braindecode

# Open the notebook in [colab](https://colab.research.google.com/drive/1a6st3wbP3pNUJw__Ys71ckfUxCbWqFIK)

# the steps 
- Install the mne and braindecode librarie
- Dowload the EEG data from physionet
- Extract trials, only using EEG channels
- Convert data from volt to millivolt
- Make the train set and the valid set
- Define the model (in my case ShallowFBCSPNet from braindecode)
- Determine the predictions per input/trial
- Define the iterator, optimizer, epochs and the scheduler
- Train!!! 
- Test!!!

## Install the mne and braindecode librarie
```python

  !pip install mne
  !pip install braindecode
  
```
## Dowload the EEG data from physionet

before downloading the data please read :
https://mne.tools/dev/generated/mne.datasets.eegbci.load_data.html <--- for know how dowload
https://physionet.org/content/eegmmidb/1.0.0/ <--- for know what dowload 

a litle resume 

 (open and close the left or right fist)               ---> 3,7,11 
 (imagine opening and closing the left or right fist)  ---> 4,8,12 <--- I chose this
 (open and close both fists or both feet)              ---> 5,9,13
 (imagine opening and closing both fists or both feet) ---> 6,10,14

```python
  # extract the data using the function mne.datasets.eegbci.load_data()
  physionet_paths = [mne.datasets.eegbci.load_data(sub_id,[4,8,12]) for sub_id in range(1,80)]
  
  # concatenate 
  physionet_paths = np.concatenate(physionet_paths)

  # Load each of the files
  parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto',verbose='WARNING')
           for path in physionet_paths]

  # Concatenate them
  raw = concatenate_raws(parts)
  
  # Find the events in this dataset
  events, _ = mne.events_from_annotations(raw) #<--- no use the event_id for that is the "_"

  # Use only EEG channels
  eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                     exclude='bads')
```
 Each annotation includes one of three codes (T0=1, T1=2, or T2=3):
 T0 corresponds to rest
 T1 corresponds to onset of motion (real or imagined) of
 the left fist (in runs 3, 4, 7, 8, 11, and 12)
 both fists (in runs 5, 6, 9, 10, 13, and 14)
 T2 corresponds to onset of motion (real or imagined) of
 the right fist (in runs 3, 4, 7, 8, 11, and 12)
 both feet (in runs 5, 6, 9, 10, 13, and 14)

## Extract trials, only using EEG channels

```python
  epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1, proj=False, picks=eeg_channel_inds,
                  baseline=None, preload=True)
```

## Convert data from volt to millivolt
```python
  # Pytorch expects float32 for input and int64 for labels.
  X = (epoched.get_data() * 1e6).astype(np.float32)
  y = (epoched.events[:,2] - 2).astype(np.int64) #this convert the labels left=2 and right=3 to 0 and 1 respectively
```
## Make the train set and the valid set
using the fuction SignalAndTarget from braindecode we can make the train and valid set

```python
  train_set = SignalAndTarget(X[:2500], y=y[:2500]) 
  valid_set = SignalAndTarget(X[2700:3000], y=y[2700:3000]) 
```
## Define the model
### the arquitecture

![shallow](https://github.com/DavidSilveraGabriel/Using-mne-and-braindecode/blob/master/img/shallow%20img.png)


## Determine the predictions per input/trial

## Define the iterator, optimizer, epochs and the scheduler

## Train!!! 

## Test!!!
