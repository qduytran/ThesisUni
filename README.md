# ThesisUni
[Quang-Duy Tran]( https://github.com/qduytran), [UET-VNU]( https://uet.vnu.edu.vn/)

This repository is where the code for my full-time university graduation thesis is stored, my topic is related to **methods of processing EEG signals**, specifically the analysis of **physiological information** in the frequency domain.

**Improving the Estimation of the 1/f Component in EEG Power Spectrum Parameterization for Alzheimer's Disease Diagnosis** [[``Click this’]]( https://drive.google.com/drive/folders/1A7miuQuSXcl0xZv36eetDHszSRXDuPvf?usp=sharing) 
<p align="center">
  <img src="assets/workflow.png" alt="Workflow" />
</p>
<p align="center">
  <strong>Overview</strong>
</p>

## Installation
The code requires `python enviroment`, you need to install `FOOOF` toolbox [here]( https://fooof-tools.github.io/fooof/).
Besides, you need to install libraries such as **mne, scipy** to process EEG signals on the frequency domain.

## Quick Start
This thesis works in the spectral domain. You should calculate the spectral first through power spectrum density estimation method by the file `welch.py`. Using python documents or matlab command `doc pwelch` to check it’s code and tutorials.

Prior to applying the ``FOOOF`` spectral decomposition method, specifically to separate the periodic (``oscillatory peaks``) and aperiodic (``1/f`` component) components, we performed a flat spectrum analysis in the frequency range of ``0`` to ``50`` Hz to determine the appropriate frequency range for model fitting and spectral estimation. 

Then, using the main function to run workflow followed by **overview** picture:
```
python main.py
```

It is recommended to adjust the data paths to align with the current working environment. The main function generates `.csv` files corresponding to each data group, which are later employed for tasks such as model estimation, classification, and further analytical procedures.

## Parameter settings
Note that the parameters should be adjusted according to your specific dataset. In our implementation, different sets of parameters are configured for each step illustrated in the **overview** figure.

## Results


## Expansion
- **`calculate_average_psd.py`**  
  Computes the average and variance of Power Spectral Density (PSD) across all subjects in each group (AD or CN), separately for each EEG channel.

- **`FOOOF.ipynb`**  
  Performs spectral decomposition using the FOOOF algorithm. Evaluates model fitting using R² and MAE metrics for periodic components.

- **`plot_error_r2.py`, `plot_r2_mae.py`, `plot_mae_r2.py`**  
  Generate different visualizations to evaluate the accuracy of spectral estimation. See the code or contact the authors for further details.

- **`plot_topo.py`**  
  Plots topographic maps of aperiodic components with and without flat-spectrum estimation, highlighting the effect of preprocessing on spectral accuracy.

- **`classify.py`**  
  Loads `.csv` files and applies basic machine learning models to classify subjects as AD or CN.

- **`plot_all_roc.py`**  
  Draws ROC curves for classification results across different FOOOF-based spectral estimation settings.

## Contributors
Quang-Duy Tran (quangduytran812@gmail.com), Nguyen Linh Trung  (linhtrung@vnu.edu.vn), Le Quoc Anh (lqanh@vnu.edu.vn), Signals and Systems Laboratory, FET-UET-VNU.

## Acknowledgements
This work was supported by the members of Signals and Systems Laboratory. Data is taken on open source from [Open Neuron]( https://github.com/OpenNeuroDatasets/ds004504).


