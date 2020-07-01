rm -r /media/data/translagent/saved_results_NEW
mkdir /media/data/translagent/saved_results_NEW
rm log.train
nvidia-smi
nohup python ./src/sentence/nmt.py --dataset multi30k --decode_how beam --src de --trg en --task 1 1>log.train &
#nohup python ./src/sentence/train_seq_joint.py --dataset multi30k --task 2 1>log.train &
#nohup python ./src/sentence/nmt.py --dataset multi30k --task 2 --src en --trg de 1>log.train &

