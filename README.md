# ECNMT: Emergent Communication Pretraining for Few-Shot Machine Translation
This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Edoardo Maria Ponti, Ivan Vulić, and Anna Korhonen. 2020. *Emergent Communication Pretraining for Few-Shot Machine Translation*. In Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020).

The model is a form of unsupervised knowledge transfer in the absence of linguistic data, where a model is first pre-trained on *artificial* languages emerging from referential games and then fine-tuned on few-shot downstream tasks like neural machine translation.

![Emergent Communication and Machine Translation](model.png "Emergent Communication and Machine Translation")

## Dependencies

- PyTorch 1.3.1
- Python 3.6

## Data and Pretrained Models
The original data sets used in our project include [MS COCO](http://cocodataset.org/#home) for Emergent Communication Pretraining, and [Multi30k Task 1](https://github.com/multi30k/dataset) and [Europarl](http://www.statmt.org/europarl/v7/) for NMT fine-tuning. Text preprcessing is based on [Moses](https://github.com/moses-smt/mosesdecoder "Moses") and [Subword-NMT](https://github.com/rsennrich/subword-nmt "Subword-NMT"). Please cite them accordingly.

To download COCO image features and preprocessed EN-DE (DE-EN) data, please follow the [Translagent](https://github.com/facebookresearch/translagent) repo, where image features are located in "./coco_new/half_feats/" and EN-DE (DE-EN) data are in "./multi30k_reorg/task1/".

The preprocessed data for EN-CS (CS-EN), EN-RO (RO-EN) and EN-FR (FR-EN), and EC pertrained models for all four language pairs can be downloaded via a Google Drive [link](https://drive.google.com/drive/folders/1sMWfvfRf9uj-LJJTye7XCixsES1EPslr?usp=sharing.). Please refer to the separate readme file therein for more details. 
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
