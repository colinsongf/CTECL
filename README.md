
# CTECL

**CTECL**, or **C**hinese **T**extual **E**ntailment **C**hunks And **L**abels.

This is the code for the paper "A Recognizing Research Based on Deep Learning For Chinese Textual Entailment Chunks And Labels"

# Usage

Firstly, you should download the Chinese Simplified Model from [BERT](https://github.com/google-research/bert)

## Seven-category entailment type recognion

```

export BERT_BASE_DIR=/YOURPATH/chinese_L-12_H-768_A-12
export MY_DATASET=/DATADIR
python3 run_classifier.py --use_tpu=False --task_name=MCE --do_train=true \
    --do_eval=true --do_predict=true --data_dir=$MY_DATASET \
    --vocab_file=$BERT_BASE_DIR/vocab.txt  \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 
    --train_batch_size=16 --learning_rate=3e-5 --num_train_epochs=2.0 \
    --output_dir=YOUR_OUTPUT_DIR
```

## Entailment Chunk Boundary Recognion


For seventeen-category entailment chunk boundary recognion, you can run your code by using

```
export checkpoint=/YOURPATH/chinese_L-12_H-768_A-12
export NERdata=/DATADIR

python3 mce19.py --task_name=MCE  --do_train=True  --do_eval=True \
    --do_predict=True --data_dir=$NERdata --vocab_file=$checkpoint/vocab.txt \
    --bert_config_file=$checkpoint/bert_config.json \
    --init_checkpoint=$checkpoint/bert_model.ckpt --max_seq_length=128 \
    --train_batch_size=16 --learning_rate=2e-5   --num_train_epochs=3 \
    --output_dir=YOUR_OUTPUT_DIR
```
