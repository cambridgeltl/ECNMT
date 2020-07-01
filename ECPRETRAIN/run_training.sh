rm -r /media/data/coco/saved_results_ECPRETRAIN
rm log.train
nvidia-smi
nohup python ./src/sentence/train_seq_joint.py --dataset coco --decode_how beam 1>log.train &

