# Brain-Computer Interfaces (BCIs) for Motor Imagery Classification

![Brain](https://cdn.technologynetworks.com/tn/images/thumbs/webp/640_360/first-ever-non-invasive-brain-computer-interface-developed-320941.webp?v=10240476)

## Introduction

Brain-Computer Interfaces (BCIs) are a transformative technology that enables direct communication between the brain and external devices. They achieve this by acquiring and processing brain signals, primarily using methods like Electroencephalography (EEG), to interpret a user's intentions. The core objective of BCI is to create a new channel for voluntary control by recognizing different mental states. These mental states, often represented as distinct patterns in EEG data, are then classified by machine learning systems. Signal processing is crucial, with feature extraction and selection methods applied to identify the most informative aspects for accurate classification. The proper selection of these features is paramount in the design of an effective BCI.

## Project Overview: Using MNE and Braindecode

This mini-project explores the application of the [mne](https://mne.tools/dev/index.html) and [braindecode](https://robintibor.github.io/braindecode/index.html) libraries to analyze and classify publicly available EEG motor imagery (MI) data from [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/). We will leverage deep convolutional neural networks (ConvNets) as described in the [Braindecode paper](https://arxiv.org/pdf/1703.05051.pdf) to decode imagined hand movements from raw EEG data.

**[Open this notebook in Google Colab](https://colab.research.google.com/drive/1a6st3wbP3pNUJw__Ys71ckfUxCbWqFIK)**

## Objective

The goal of this project is to perform classification of EEG data associated with motor imagery in the motor cortex. Specifically, we'll classify between imagined movements of the left and right hands.

## Project Steps

1.  **Install Necessary Libraries**: `mne` and `braindecode`
2.  **Download EEG Data**: Acquire data from Physionet's EEG Motor Movement/Imagery Database.
3.  **Extract Trials**: Isolate EEG segments corresponding to specific motor imagery events.
4.  **Data Preprocessing**: Convert EEG voltage data from volts to millivolts.
5.  **Create Train and Validation Sets**: Prepare datasets for model training and evaluation.
6.  **Model Definition**: Implement a ShallowFBCSPNet model from `braindecode`.
7.  **Training Setup**: Define iterator, optimizer, training epochs and learning rate scheduler.
8.  **Model Training**: Train the model using the prepared datasets.
9.  **Model Evaluation**: Assess the model's performance on a held-out test set.

### 1. Install Libraries

```python
!pip install mne
!pip install braindecode
```

### 2. Download EEG Data

**Important**: Before downloading, please refer to the documentation:

*   [MNE Documentation on Loading EEGBCI Data](https://mne.tools/dev/generated/mne.datasets.eegbci.load_data.html)
*   [Physionet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/)

**Dataset Overview:**

The dataset contains EEG recordings of subjects performing various motor tasks:

*   **Tasks we are using:**
    *   (3, 7, 11) Real left or right fist movement
    *   **(4, 8, 12) Imagined left or right fist movement (We are using this)**
    *   (5, 9, 13) Real both fists or feet movement
    *   (6, 10, 14) Imagined both fists or feet movement

```python
import mne
import numpy as np
from braindecode.datasets import SignalAndTarget
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import ShallowFBCSPNet
from braindecode.training import (
    CroppedLoss,
    ExponentialScheduler,
    CropsFromTrialsIterator
)
from torch.optim import AdamW
import torch
from braindecode.util import set_random_seeds
from torch.nn.functional import square, log, log_softmax
from braindecode.preprocessing import preprocess, create_fixed_length_windows
from braindecode.datautil import load_concat_dataset
from braindecode.training import Trainer, trial_preds_from_window_preds
from braindecode.datasets import BaseDataset, create_from_mne_raw, create_from_mne_epochs
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring
from torch.optim.lr_scheduler import CosineAnnealingLR

from mne.io import concatenate_raws
set_random_seeds(seed=1, cuda=False)
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

### 3. Extract Trials
```python
epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1, proj=False, picks=eeg_channel_inds,
baseline=None, preload=True)
```

### 4. Data Preprocessing
```python
# Pytorch expects float32 for input and int64 for labels.
X = (epoched.get_data() * 1e6).astype(np.float32)
y = (epoched.events[:,2] - 2).astype(np.int64) #this convert the labels left=2 and right=3 to 0 and 1 respectively
```
### 5. Create Train and Validation Sets
```python
train_set = SignalAndTarget(X[:2500, y=y[:2500) 
valid_set = SignalAndTarget(X[2700:3000], y=y[2700:3000]) 
```
### 6. Define the Model
```python
n_channels = X.shape[1
input_time_length = X.shape[2]
model = ShallowFBCSPNet(n_channels,2,input_time_length=input_time_length, final_conv_length='auto')
```

**ShallowFBCSPNet Architecture:**

![shallow](https://github.com/DavidSilveraGabriel/Using-mne-and-braindecode/blob/master/img/shallow%20img.png)

```text
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
  (softmax): LogSoftmax(dim=1)
  (squeeze): Expression(expression=_squeeze_final_output)
)
```

### 7. Training Setup
```python
n_preds_per_input = 1
n_updates_per_epoch = len(train_set) // 64
iterator = CropsFromTrialsIterator(batch_size=64,input_time_length=input_time_length,
n_preds_per_input=n_preds_per_input)
optimizer = AdamW(model.parameters(), lr=0.06255 * 0.015, weight_decay=0)
n_epochs = 50
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * n_updates_per_epoch)
```

### 8. Model Training
```python
trainer = Trainer(
    model=model,
    iterator=iterator,
    optimizer=optimizer,
    loss_function=CroppedLoss(),
    callbacks=[
       EpochScoring(
           scoring="accuracy",
           lower_is_better=False,
           on_train=True,
           name="train_accuracy",
           ),
       EpochScoring(
           scoring="accuracy",
           lower_is_better=False,
           on_train=False,
           name="valid_accuracy",
           ),
    ,
)
losses = []
for epoch in range(n_epochs):
    for batch, trial_i in iterator(train_set,return_ids=True):
        batch = batch.to('cpu')
        optimizer.zero_grad()
        preds = model(batch)
        loss = trainer.get_loss(preds, batch, y=train_set.y[trial_i])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    #validation
    model.eval()
    batch, trial_i = next(iterator(valid_set,return_ids=True))
    batch = batch.to('cpu')
    with torch.no_grad():
        preds = model(batch)
        loss = trainer.get_loss(preds, batch, y=valid_set.y[trial_i])
    #test
    batch_test , trial_i_test = next(iterator(SignalAndTarget(X[3000:3200], y[3000:3200]),return_ids=True))
    batch_test = batch_test.to('cpu')
    with torch.no_grad():
        preds_test = model(batch_test)
        loss_test = trainer.get_loss(preds_test, batch_test, y=SignalAndTarget(X[3000:3200], y[3000:3200]).y[trial_i_test])
        trial_pred = trial_preds_from_window_preds(preds_test, batch_test, trial_i_test)
        acc_test = np.mean(trial_pred.argmax(1) == SignalAndTarget(X[3000:3200], y[3000:3200]).y[trial_i_test])
    model.train()
    print(f"Epoch {epoch}")
    print(f"Train  Loss: {np.mean(losses[-n_updates_per_epoch:):.5f}")
    print(f"Train  Accuracy: {trainer.callbacks[0].history[-1]['train_accuracy']:.1%}")
    print(f"Valid  Loss: {loss:.5f}")
    print(f"Valid  Accuracy: {trainer.callbacks[1].history[-1]['valid_accuracy']:.1%}")
```

### 9. Model Evaluation

```python
print(f"Test Loss: {loss_test:.5f}")
print(f"Test Accuracy: {acc_test:.1%}")
```

**Observed Results:**

```text
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
Test Loss: 0.59670
Test Accuracy: 70.5%
```

The results indicate a clear **overfitting** issue, where the model performs exceptionally well on the training data but struggles to generalize to unseen validation and test data.

## Conclusions

Brain-Computer Interfaces hold tremendous promise for improving the lives of individuals with severe disabilities, such as those who suffer from motor impairments. The ability to control external devices using brain signals offers a path toward enhanced autonomy and independence for these individuals. This mini-project has provided hands-on experience with the intricacies of EEG signal processing and the application of machine learning techniques for BCI development. It also reinforces the need for rigorous methods to mitigate overfitting for better model generalization. The prospect of advancing this field and contributing to the well-being of people with disabilities provides a powerful motivation for future research.

## References

*   [Braindecode GitHub Repository](https://github.com/braindecode/braindecode)
