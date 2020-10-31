# ECNMT: Emergent Communication Pretraining for Few-Shot Machine Translation
This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Edoardo Maria Ponti, Ivan Vulić, and Anna Korhonen. 2020. *Emergent Communication Pretraining for Few-Shot Machine Translation*. In Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020).

## Dependencies

- PyTorch 1.3.1
- Python 3.6

## Data and Pretrained Models
The data and pertrained models can be downloaded via a Google Drive link (TBA). Please refer to the separate readme file therein for more details.

## Experiment

Step 1: run EC pretraining (otherwise go to Step 2 and use a pretrained model).
```bash
cd ./ECPRETRAIN
sh run_training.sh
 ```
                         
Step 2: run NMT fine-tuning (please modify the roots for training data, pretrained model and saved path before).
```bash
cd ./NMT
sh run_training.sh
```

Optional: run baseline

```bash
cd ./BASELINENMT
sh run_training.sh
 ```
   
## Citation

    @inproceedings{YL:2020,
      author    = {Yaoyiran Li and Edoardo Maria Ponti and Ivan Vulić and Anna Korhonen},
      title     = {Emergent Communication Pretraining for Few-Shot Machine Translation},
      year      = {2020},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
    }
    
## Acknowledgements

Part of the code is based on https://github.com/facebookresearch/translagent. Please cite it accordingly.
