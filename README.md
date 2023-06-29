# Steps

Fake News: 0
Real News: 1

Left Leaning: Privileged Group: 0
Right Leaning: Unprivileged Group: 1

TN: predicted = 0 (fake), actual = 0(fake)  
FN: predicted = 0(fake), actual = 1(real)  
FP: predicted = 1 (real), actual = 0 (fake)
FN: predicted = 1 (real), actual = 1(real) 

SPD = 0 to be fair
EOD = 0 to be fair
DIR = 1 to be fair
AOD = 0 to be fair

Debiasing using ROC
Threshold: lies between 0 and 1
Margin: lies between 0 and min(threshold, 1-threshold )

```bash

# steps to be done only the first time
pip install .
cd src

# To Train a different variant of the model or different hyperparameters, etc.
python train.py

# utility functions for metrics and debiasing
utils.py

# To test bias metrics - sklearn + aif360 
python metrics.py

# To debias (posthoc debiasing using Reject Option Classifier)
python debias_input_prepare.py
python debias_roc.py

# To perform analysis on BERT using captum
python bert_analysis.py

# To extract important tokens using SHAP
python out_of_shap.py

# To extract important tokens using LIME
python in_the_lime.py

# To extract important tokens using Integrated Gradience
python int_grad.py

# to extract important phrases using SHAP and LIME
python phrase_level_shap.py
python phrase_level_lime.py

# To extract important sentences using SHAP
python sentence_level_shap.py

# To perform data injection attacks
python freq_hamla.py, python salience_hamla.py
python injection_attack_test.py
python attack_metrics.py
```
