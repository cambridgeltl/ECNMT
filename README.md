# ECNMT
The repository is the PyTorch implementation of the following paper: 
Emergent Communication Pretraining for Few-Shot Machine Translation
Yaoyiran Li, Edoardo Maria Ponti, Ivan Vulić, and Anna Korhonen
Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020), long paper, 2020, online

Part of the code is developed based on the publicly available open-source github repo https://github.com/facebookresearch/translagent , which is the offical implementation of Emergent Translation in Multi-Agent Communication, ICLR 2018. 


ECNMT
Pytorch 1.3.1 or above
Python 3.6

Our Method, RUN:

Step 1 (EC Pretraining): 

    cd ./ECPRETRAIN
    sh run_training.sh
                         
Step 2 (NMT Fine-tuning): 

    cd ./NMT
    sh run_training.sh

Baseline, RUN:  

    cd ./BASELINENMT
    sh run_training.sh
