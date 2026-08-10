# GraphFlow Codebase

Source code for NeurIPS 2025

## Dependencies

Environment setup:

```
conda create -n graphflow python=3.10 -y
conda activate graphflow
pip install torch transformers peft trl datasets stark_qa python-liquid networkx numpy tqdm wandb
```


## Training


```bash
python graphflow_trainer_savemodel.py \
  --dataset_name prime \
  --data_root /path/to/stark_data \
  --pretrain_model_dir /path/to/Meta-Llama-3-8B-Instruct \
  --r 32 \
  --alpha 16 \
  --lr 1e-5 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --max_length 1024 \
  --num_negative 4 \
  --eval_ratio 0.5
```