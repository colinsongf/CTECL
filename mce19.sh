export checkpoint=/home/wanying/bert-mce/chinese_L-12_H-768_A-12
export NERdata=/home/wanying/bert-mce/data/

python3 mce19.py --task_name=MCE  --do_train=True  --do_eval=True --do_predict=True --data_dir=$NERdata --vocab_file=$checkpoint/vocab.txt  --bert_config_file=$checkpoint/bert_config.json          --init_checkpoint=$checkpoint/bert_model.ckpt --max_seq_length=128  --train_batch_size=16 --learning_rate=2e-5   --num_train_epochs=3       --output_dir=./mce19/
