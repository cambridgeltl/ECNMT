rm -r /media/data/translagent2/saved_results
mkdir /media/data/translagent2/saved_results
rm log.train
nvidia-smi
nohup python ./src/sentence/nmt.py --dataset multi30k --decode_how beam --src en --trg de --task 1 1>log.train &
#nohup python ./src/sentence/train_seq_joint.py --dataset multi30k --task 2 1>log.train &
#nohup python ./src/sentence/nmt.py --dataset multi30k --task 2 --src en --trg de 1>log.train &

