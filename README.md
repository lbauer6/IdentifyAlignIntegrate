# Identify, Align, and Integrate: Matching Knowledge Graphs to Commonsense Reasoning Tasks (EACL 2021)

This repository contains the code and setup instructions for our EACL 2021 paper "Identify, Align, and Integrate: Matching Knowledge Graphs to Commonsense Reasoning Tasks". See full paper [here](https://arxiv.org/abs/2104.10193).

We release our knowledge-augmented data (identify), code for our KS model (align), and our commonsense probes (integrate).   

## Data

### Knowledge-Augmented Data
We release knowledge-augmented data for SocialIQA, PhysicalIQA, and MCScript2.0. This data is augmented via the various extraction methods listed in the paper (conditioning, shape, filtering) for the best KG-to-task matches. You can find our data [here](https://drive.google.com/drive/folders/18ePzPXlv4mb14c00bJj8WXJuuBYaG6oW?usp=sharing). Move this data into `src/data/` to train models.


### Probes
We also release commonsense probes in a QA setup and an MLM setup, found in `probes/`. 

## Training 
We provide the following commands to train our KS Model on SIQA with ATOMIC with best settings (paths, CS-3, KS+) for 4 different configurations (QC-HQ, QC-HR, A-HQ, A-HR). Our training scripts are found in `src/`.

To train models for SIQA QC-HQ, run:
```
python3 run_siqa_qchq_cs3.py  \
    --task_name SIQA \
    --do_train \
    --do_eval  \
    --do_lower_case \
    --data_dir data \
    --bert_model bert-base-uncased/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \  
    --output_dir siqa_qchq_output
```

To train models for SIQA QC-HR, run:
```
python3 run_siqa_qchr_cs3.py  \
    --task_name SIQA \
    --do_train \
    --do_eval  \
    --do_lower_case \
    --data_dir data \
    --bert_model bert-base-uncased/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \  
    --output_dir siqa_qchr_output
```

To train models for SIQA A-HQ, run:
```
python3 run_siqa_ahq_cs3.py  \
    --task_name SIQA \
    --do_train \
    --do_eval  \
    --do_lower_case \
    --data_dir data \
    --bert_model bert-base-uncased/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \  
    --output_dir siqa_ahq_output
```

To train models for SIQA A-HR, run:
```
python3 run_siqa_ahr_cs3.py  \
    --task_name SIQA \
    --do_train \
    --do_eval  \
    --do_lower_case \
    --data_dir data \
    --bert_model bert-base-uncased/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \  
    --output_dir siqa_ahr_output
```


We provide the following commands to train our KS Model on PIQA with ConceptNet (subgraphs, CS-3, KS+) for 4 different configurations (QC-HQ, QC-HR, A-HQ, A-HR).

To train models for PIQA QC-HQ, run:
```
python3 run_multiple_choice.py \
    --seed 1 \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name PIQAKS \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/piqa_data/cn_data/piqa_data_cn_qc_hq_subgraph/ \
    --max_seq_length 150 \
    --per_gpu_eval_batch_size=6 \
    --per_gpu_train_batch_size=6 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --output_dir piqa_qchq_output
```

To train models for PIQA QC-HR, run:
```
python3 run_multiple_choice.py \
    --seed 1 \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name PIQAKS \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/piqa_data/cn_data/piqa_data_cn_qc_hr_subgraph/ \
    --max_seq_length 150 \
    --per_gpu_eval_batch_size=6 \
    --per_gpu_train_batch_size=6 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --output_dir piqa_qchr_output
```


To train models for PIQA A-HQ, run:
```
python3 run_multiple_choice.py \
    --seed 1 \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name PIQAKS \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/piqa_data/cn_data/piqa_data_cn_a_hq_subgraph/ \
    --max_seq_length 150 \
    --per_gpu_eval_batch_size=6 \
    --per_gpu_train_batch_size=6 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --output_dir piqa_ahq_output
```


To train models for PIQA A-HR, run:
```
python3 run_multiple_choice.py \
    --seed 1 \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name PIQAKS \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/piqa_data/cn_data/piqa_data_cn_a_hr_subgraph/ \
    --max_seq_length 150 \
    --per_gpu_eval_batch_size=6 \
    --per_gpu_train_batch_size=6 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --output_dir piqa_ahr_output
```

## Reference
If you find this code helpful, please consider citing the following paper:

```
@inproceedings{bauer2021identify,
  title={Identify, Align, and Integrate: Matching Knowledge Graphs to Commonsense Reasoning Tasks},
  author={Lisa Bauer and Mohit Bansal},
  booktitle={Proceedings of the European Chapter of the Association for Computational Linguistics},
  year={2021}
}
```
