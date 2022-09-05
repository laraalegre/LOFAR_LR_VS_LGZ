# LOFAR LR vs LGZ

Predict which sources of LoTSS can be cross-matched by LR or require visual analysis

Paper: https://doi.org/10.1093/mnras/stac1888

Outline
------------

    ├── README.md 
    │
    ├── data (untracked)
    │   └── catalogues     
    │       ├── originals   <- Original full catalogues for sources and Gaussians (a,b)
    │       ├── pybdsf      <- PyBDSF properties for sources and Gaussians (a)
    │       └── lr          <- LR for sources and Gaussians (b)
    │ 
    ├── scripts
    │       └── predict     <- Code to create the features and output the predictions
    │
    ├── models   
    │      └──  gbc         <- Trained model weights for the GBC model
    │
    ├── results 
    │      └──  gbc         <- Predictions example for LoTSS DR2 - P21   
    │
    └── requirements        <- Python packages 


## INPUTS 

### LoTSS DR2 Catalogues

a) The original full catalogues for sources and Gaussians can be found at https://lofar-surveys.org/dr2_release.html with the names:
- LoTSS_DR2_v110_masked.srl.fits
- LoTSS_DR2_v110.gaus.fits

  The PyBDSF catalogues for sources and Gaussians can also be produced using:
  - code: http://ascl.net/1502.007; https://www.astron.nl/citt/pybdsf/
  - Paper where PYBDSF for DR1 is described:https://doi.org/10.1051/0004-6361/201833559
  - Notes: Compute PyBDSF properties of sources and Gaussians (essential: Maj, Min, Peak Flux, Total Flux).


b) LR for sources and Gaussians can be found internatlly on Herts with the names:
- LoTSS_DR2_v100.gaus_13h.lr-full.fits
- LoTSS_DR2_v100.srl_13h.lr-full.sorted_step3_flux4.fits

  The Likelihood Ratios (LR) for sources and Gaussians can also be calculated using:
  - Code: https://github.com/nudomarinero/lr_lotss_dr2 
  - Paper describing the LR for DR1: https://doi.org/10.1051/0004-6361/201833564
  - Notes: Compute LR values for sources and Gaussians (essential: lr and lr_dist).

### LR thresholds values used for LoTSS DR2
- 13h
  - LR_thresh = 0.309 (n, dec>=32.375) 
  - LR_thresh = 0.328 (s, dec<=32.375)
- 0h
  - LR_thresh = 0.394

## OUPUTS 

- features.fits  - list of features used to train the model which are necessary to output the predictions.  
- pred_thresholds.csv - predictions for different thresholds (we used 0.20 for DR1). 

Note: For LoTSS DR1, the list of features and the predictions (prediction_0.20) can be found as a supplementary material of the paper https://doi.org/10.1093/mnras/stac1888
