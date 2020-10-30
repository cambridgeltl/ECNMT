# ECNMT
The repository is the PyTorch implementation of the following paper: 

"Emergent Communication Pretraining for Few-Shot Machine Translation", Yaoyiran Li, Edoardo Maria Ponti, Ivan Vulić, and Anna Korhonen, Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020), long paper, 2020, online.

Part of the code is developed based on the publicly available open-source github repo https://github.com/facebookresearch/translagent , which is the offical implementation of "Emergent Translation in Multi-Agent Communication", ICLR 2018. 


Dependencies

    Pytorch 1.3.1 (or above)
    Python 3.6
    CUDA 10.1 (In our implementation the driver version is 418.87.01. We did not test other CUDA versions.)
    Numpy

Our data and pertrained models can be downloaded via Google Drive (please refer to a readme file in it):
    
    https://drive.google.com/drive/folders/1sMWfvfRf9uj-LJJTye7XCixsES1EPslr?usp=sharing 


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
