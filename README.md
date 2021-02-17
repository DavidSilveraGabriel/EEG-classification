# Introduction

![Brain](https://cdn.technologynetworks.com/tn/images/thumbs/webp/640_360/first-ever-non-invasive-brain-computer-interface-developed-320941.webp?v=10240476)

Brain Computer Interfaces (BCI) are a technology based on the
acquisition and processing of brain signals for the control of various
devices. Its main objective is to provide a new outlet channel to the
user's brain that requires voluntary adaptive control. BCIs usually focus on recognizing events that are acquired by methods like the Electroencephalogram (EEG). 
These events represent different
mental states, which should be identified as different classes by
a classification system. In BCI, after acquiring the brain signals, they are prepared for further processing. To extract
and select the characteristics various methods are applied, in which, in
Depending on the type of control signals used, it would be appropriate to identify
a subset that optimizes the tasks for their classification. The selection of
the most discriminative characteristics is essential when designing
Functional BCI.

# Using-mne-and-braindecode
in this mini project i will use the libraries [mne](https://mne.tools/dev/index.html) and [braindecode](https://robintibor.github.io/braindecode/index.html) to analyze and classify public domain EEG(ML) signals provided by [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/)
here the [paper](https://arxiv.org/pdf/1703.05051.pdf) from braindecode where is studied deep ConvNets with a range of different architectures, designed for decoding imagined or executed movements from raw EEG 

# Open the notebook in [colab](https://colab.research.google.com/drive/1a6st3wbP3pNUJw__Ys71ckfUxCbWqFIK)

# objetive

perform a classification of eeg data from the motor cortex


# the steps 
- Install the mne and braindecode librarie
- Dowload the EEG data from physionet
- Extract trials, only using EEG channels
- Convert data from volt to millivolt
- Make the train set and the valid set
- Define the model (in my case ShallowFBCSPNet from braindecode)
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
 ```
   Each annotation includes one of three codes (T0=1, T1=2, or T2=3):

   T0 corresponds to rest

   T1 --> corresponds to onset of motion (real or imagined) of
          the left fist (in runs 3, 4, 7, 8, 11, and 12)
          both fists (in runs 5, 6, 9, 10, 13, and 14)

   T2 --> corresponds to onset of motion (real or imagined) of
          the right fist (in runs 3, 4, 7, 8, 11, and 12)
          both feet (in runs 5, 6, 9, 10, 13, and 14)
```
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

    Sequential(
      (dimshuffle): Expression(expression=_transpose_time_to_spat)
      (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
      (conv_spat): Conv2d(40, 40, kernel_size=(1, 64), stride=(1, 1), bias=False)
      (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_nonlin): Expression(expression=square)
      (pool): AvgPool2d(kernel_size=(75, 1), stride=(1, 1), padding=0)
      (pool_nonlin): Expression(expression=safe_log)
      (drop): Dropout(p=0.5, inplace=False)
      (conv_classifier): Conv2d(40, 2, kernel_size=(27, 1), stride=(1, 1), dilation=(15, 1))
      (softmax): LogSoftmax()
      (squeeze): Expression(expression=_squeeze_final_output)
    )

## Define the iterator, optimizer, epochs and the scheduler

```python

    iterator = CropsFromTrialsIterator(batch_size=64,input_time_length=input_time_length,
                                      n_preds_per_input=n_preds_per_input)

    optimizer = AdamW(model.parameters(), lr=0.06255 * 0.015, weight_decay=0)

    n_epochs = 50

    scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)
```

## Train!!! 
### lets look the last 5 epochs ouputs 
```python
    Epoch 45
    Train  Loss: 0.11758
    Train  Accuracy: 98.3%
    Valid  Loss: 0.57367
    Valid  Accuracy: 77.3%
    Epoch 46
    Train  Loss: 0.11460
    Train  Accuracy: 98.6%
    Valid  Loss: 0.57288
    Valid  Accuracy: 77.0%
    Epoch 47
    Train  Loss: 0.11470
    Train  Accuracy: 98.6%
    Valid  Loss: 0.57359
    Valid  Accuracy: 77.3%
    Epoch 48
    Train  Loss: 0.11501
    Train  Accuracy: 98.6%
    Valid  Loss: 0.57888
    Valid  Accuracy: 76.7%
    Epoch 49
    Train  Loss: 0.11461
    Train  Accuracy: 98.5%
    Valid  Loss: 0.57351
    Valid  Accuracy: 77.0%
```
## Test!!!
### lets print the ouput of the test 

```python

    Test Loss: 0.59670
    Test Accuracy: 70.5%

```
clearly, the model is overfitted, not being able to generalize the test and validation data well enough

# Conclutions 

The benefits of the brain computer interface (BCI) in the lives of people with disabilities can be enormous, imagine that a person in a wheelchair with dystonia, multiple sclerosis, or another disability that does not allow them to move even their arms To use a conventional wheelchair, this person is totally enclosed, lacking freedom in an unmovable body. reason why their life becomes much more difficult, that is why the development of BCI technologies are essential for these people
Now my work here on this mini project has provided me with a deeper understanding of electroencephalographic signals and their classification for later use in BCIs. It has made me reflect on the importance of investing in the development and research of everything related to this area of computer science, the great impact that can be obtained by the development of such technologies, and finally it has given me the objective of contribute my grain of sand so that one day the people who suffer from these horrible disabilities can lead a normal life despite everything

## References

https://github.com/braindecode/braindecode

image --> https://www.technologynetworks.com/informatics/news/first-ever-non-invasive-brain-computer-interface-developed-320941

https://mne.tools/stable/index.html

https://braindecode.org/
