# Part of the code is developed based on the publicly available open-source github repo https://github.com/facebookresearch/translagent , which is the offical implementation of Emergent Translation in Multi-Agent Communication, ICLR 2018. 
# The code we submit is only for COLING 2020 double-blind review where reviewers could better understand how we implement our proposed method. Any individual is not allowed to keep, re-develop or distribute the code we submit.
# ECNMT
ECNMT
Pytorch 1.3.1 or above
Python 3.6

Our Method, RUN:
Step 1 (EC Pretraining): cd ./ECPRETRAIN

                         sh run_training.sh
                         
Step 2 (NMT Fine-tuning): cd ./NMT

                          sh run_training.sh

Baseline, RUN:  cd ./BASELINENMT

                sh run_training.sh
