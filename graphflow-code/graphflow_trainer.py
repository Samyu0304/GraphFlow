from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
import wandb
import os
import argparse
from stark_qa import load_qa, load_skb
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import json
import tqdm
# from dataset import GFlowNetPathDataset, data_collator_fn
from dataset_subtraj import GFlowNetPathStepDataset, data_collator_fn
from model import GFlowLlama
import random


class GFlowNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize WandB
        # wandb.init(project="gflownet-project", entity="your_wandb_username")
    
    def training_step(self, model, inputs, *args, **kwargs):
        print("Model device (inside training_step):", next(model.parameters()).device)
        return super().training_step(model, inputs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # device = inputs["state_input_ids"].device

        state_input_ids = inputs["state_input_ids"]#.to(device)
        state_attention_mask = inputs["state_attention_mask"]#.to(device)
        neighbor_input_ids = inputs["neighbor_input_ids"]#.to(device)
        neighbor_attention_mask = inputs["neighbor_attention_mask"]#.to(device)
        next_state_input_ids = inputs["next_state_input_ids"]#.to(device)
        next_state_attention_mask = inputs["next_state_attention_mask"]#.to(device)
        
        actions = inputs["actions"]#.to(device)
        mask = inputs["mask"]#.to(device)
        reward = inputs["reward"]

        log_flow, log_next_flow, policy_probs = model(
            state_input_ids,
            state_attention_mask,
            next_state_input_ids,
            next_state_attention_mask,
            neighbor_input_ids,
            neighbor_attention_mask,
            actions,
            mask,
            reward,
        )
        
        print(policy_probs, actions)
        
        # flow is the log value, next _flow is the log value log(F(S))
        # treat
        eps = 1e-8
        B = log_flow.size(0)

        # ==== 1. 获取选中动作的概率 ====
        action_probs = policy_probs[torch.arange(B), actions]  # shape (B,)
        log_policy = torch.log(action_probs + eps)             # log π(a|s)

        # ==== 2. 主损失 ====
        balance_error = log_flow - log_policy - log_next_flow
        balance_loss = (balance_error ** 2).mean()

        # ==== 3. 边界条件损失 ====
        start_mask = (reward > 1).float()
        end_mask = (reward < 1).float()

        # 目标值是 log(F) = 1，而不是 0
        start_loss = ((log_flow - 1) ** 2 * start_mask).mean()
        end_loss = ((log_next_flow - 1) ** 2 * end_mask).mean()

        # ==== 4. 合并 ====
        loss = balance_loss + start_loss + end_loss
        print(loss, balance_loss, start_loss, end_loss)


        # Log loss to WandB
        if wandb.run is not None:
            wandb.log({"loss": loss.item(), "balance_loss": balance_loss.item(), "start_loss": start_loss.item(), "end_loss": end_loss.item()})

        return (loss, {"loss": loss,  "balance_loss": balance_loss, "start_loss": start_loss, "end_loss": end_loss}) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        model = self.model.eval()
        dataloader = self.get_eval_dataloader(eval_dataset)

        correct = 0
        total = 0
        total_subtraj_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # === 取数据并放到设备上 ===
                # device = model.device
                state_input_ids = batch["state_input_ids"]#.to(device)
                state_attention_mask = batch["state_attention_mask"]#.to(device)
                neighbor_input_ids = batch["neighbor_input_ids"]#.to(device)
                neighbor_attention_mask = batch["neighbor_attention_mask"]#.to(device)
                next_state_input_ids = batch["next_state_input_ids"]#.to(device)
                next_state_attention_mask = batch["next_state_attention_mask"]#.to(device)
                actions = batch["actions"]#.to(device)
                rewards = batch["reward"]#.to(device)
                masks = batch["mask"]#.to(device)

                # === 模型输出 ===
                log_flow, log_next_flow, policy_probs = model(
                    state_input_ids,
                    state_attention_mask,
                    next_state_input_ids,
                    next_state_attention_mask,
                    neighbor_input_ids,
                    neighbor_attention_mask,
                    actions,
                    masks,
                    rewards,
                )

                # === 准确率统计 ===
                pred_actions = policy_probs.argmax(dim=-1)
                correct += ((pred_actions == actions) * masks.bool()).sum().item()
                total += masks.sum().item()

                # === SubTB Loss ===
                eps = 1e-8
                B = log_flow.size(0)

                action_probs = policy_probs[torch.arange(B), actions]
                log_policy = torch.log(action_probs + eps)

                balance_error = log_flow - log_policy - log_next_flow
                balance_loss = (balance_error ** 2).mean()

                start_mask = (rewards > 1).float()
                end_mask = (rewards < 1).float()
                start_loss = ((log_flow - 1) ** 2 * start_mask).mean()
                end_loss = ((log_next_flow - 1) ** 2 * end_mask).mean()

                subtraj_loss = balance_loss + start_loss + end_loss

                total_subtraj_loss += subtraj_loss.item()
                num_batches += 1

        policy_accuracy = correct / total if total > 0 else 0
        avg_subtraj_loss = total_subtraj_loss / num_batches if num_batches > 0 else 0

        # Log evaluation metrics to WandB
        if wandb.run is not None:
            wandb.log({
                f"{metric_key_prefix}_policy_accuracy": policy_accuracy,
                f"{metric_key_prefix}_subtraj_loss": avg_subtraj_loss,
            })

        output[f"{metric_key_prefix}_policy_accuracy"] = policy_accuracy
        output[f"{metric_key_prefix}_subtraj_loss"] = avg_subtraj_loss
        return output
def merge_data(fnames, depth_cutoff=5, eval_ratio = 1.0):
    
    list_path_question = []
    
    print(f"total {len(fnames)} samples")
    
    for fname in fnames:
    
        with open(fname, 'r') as f:
            item = json.load(f)
        
        if item['path'] == []:
            print(f'Ignore file {fname}')
            continue

        question = item.get('qestion', item.get('question'))
        
        path_list = item['path'][:1]
        for path_item in path_list:
            if int(path_item["distance"]) >= depth_cutoff:
                continue
            elif int(path_item["distance"]) <= 1:
                continue
            else:
                list_path_question.append({'question':question, 'path':path_item['path']})
    
    num_list_path_question = len(list_path_question)
    
    if eval_ratio < 0.99:
        sampled_num = int(eval_ratio * num_list_path_question)
        list_path_question = random.sample(list_path_question, sampled_num)
        
    print(f"total {len(list_path_question)} data")
    
    
    return list_path_question
    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_root", type=str, default='/fs-computility/ai4sData/yujunchi/yujunchi/stark_data')
    parser.add_argument("--dataset_name", type=str, default="prime")
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--pretrain_model_dir", type=str, default="/fs-computility/ai4sData/yujunchi/yujunchi/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5) # 1e-5 seems not converge
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--doc_cutoff", type=int, default=400)
    parser.add_argument("--num_negative", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--depth_cutoff", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=1024) # 4096
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    data_root = args.data_root
    pretrain_model_dir = args.pretrain_model_dir
    shortest_path_dir = f'/fs-computility/ai4sData/yujunchi/yujunchi/G-Retriever-main/dataset/stark/{dataset_name}/cached_paths'
    max_length = args.max_length
    num_negative = args.num_negative
    doc_cutoff = args.doc_cutoff
    accumulation_steps = args.accumulation_steps
    
    lora_r = args.r
    lora_alpha = args.alpha
    lora_dropout = args.lora_dropout
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    eval_ratio = args.eval_ratio
    save_model_dir = os.path.join('gf_ckpt', f'{dataset_name}', f"{lora_r}_{lora_alpha}_{lr}_{n_epochs}_{batch_size}_{accumulation_steps}_{max_length}_{eval_ratio}_{num_negative}")
    
    ## load dataset split

    # load skb stark_qa
    skb_dataset = load_skb(dataset_name, root = data_root, download_processed=True)
    qa_dataset = load_qa(dataset_name, root = data_root)

    # get index split
    dataset_splits = qa_dataset.get_idx_split()
    train_index = dataset_splits['train']
    eval_index = dataset_splits['val']

    # load pre-extracted shortest path .json file
    train_fnames = []
    eval_fnames = []
    for index in train_index:
        shortest_path_json_file = f'{shortest_path_dir}/{index}.json'
        if os.path.exists(shortest_path_json_file):
            train_fnames.append(shortest_path_json_file)
    
    for index in eval_index:
        shortest_path_json_file = f'{shortest_path_dir}/{index}.json'
        if os.path.exists(shortest_path_json_file):
            eval_fnames.append(shortest_path_json_file)

    ## load path json and merge into 
    
    train_data = merge_data(train_fnames)
    eval_data = merge_data(eval_fnames, eval_ratio = eval_ratio)
    
    reward_model = AutoModelForCausalLM.from_pretrained(
            pretrain_model_dir,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,  # 加速 + 内存优化
        )
    
    model = GFlowLlama(llm_backbone=reward_model, use_lora=True, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    # model = model.to(torch.cuda.current_device())
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # model.to(local_rank)
    # print(next(model.parameters()).device)
    
    reward_tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
    reward_tokenizer.padding_side = 'left'
    reward_tokenizer.truncation_side = 'left'
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        # reward_tokenizer.pad_token = "<|end_of_text|>" # 128001
    if reward_model.config.pad_token_id is None:
        # reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        reward_model.config.pad_token_id = -1
    ## load dataset
    train_set = GFlowNetPathStepDataset(train_data, skb_dataset, reward_tokenizer, max_length = max_length, num_negatives= num_negative, doc_cutoff = doc_cutoff)
    eval_set = GFlowNetPathStepDataset(eval_data, skb_dataset, reward_tokenizer, max_length = max_length, num_negatives= num_negative, doc_cutoff = doc_cutoff)     
    
    # accelerator = Accelerator()
    # model = accelerator.prepare(model)
    # train_set = accelerator.prepare(train_set)
    # eval_set = accelerator.prepare(eval_set)   
    
    
    # Training Function (train_gflownet)
    def train_gflownet(model, train_dataset, eval_dataset, training_args):
        trainer = GFlowNetTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator_fn,
        )

        # Start training
        trainer.train()

        # Save model after training
        # save_model(model, save_dir=training_args.output_dir)

        # Optionally, evaluate the model
        eval_results = trainer.evaluate(eval_dataset)
        print(f"Evaluation results: {eval_results}")

    # Optimized TrainingArguments for GFlowNet (with bfloat16 precision)
    training_args = TrainingArguments(
        output_dir=save_model_dir,              # Output directory
        evaluation_strategy="steps",         # Evaluation steps
        eval_steps = 20,                    # eval steps
        save_strategy="best",                # Save every epoch
        learning_rate=lr,                  # Learning rate
        per_device_train_batch_size=batch_size,      # Batch size per device during training
        per_device_eval_batch_size=batch_size,       # Batch size per device during evaluation
        weight_decay=0.01,                   # Strength of weight decay
        logging_dir="./logs",                # Directory for storing logs
        logging_steps=10,                    # Log every 10 steps
        num_train_epochs=3,                  # Number of training epochs
        save_total_limit=3,                  # Only save the last 2 checkpoints
        bf16=True,                           # Enable bfloat16 precision
        gradient_accumulation_steps=accumulation_steps,       # Accumulate gradients over 2 steps
        report_to="none",                   # Report metrics to WandB
        load_best_model_at_end=True,         # Load the best model when finished training
        metric_for_best_model="eval_subtraj_loss",   # Metric for best model selection
        greater_is_better=False,             # Lower eval_loss is better
        dataloader_num_workers=1,            # Number of workers for data loading
        local_rank=os.getenv("LOCAL_RANK", -1),
        ddp_find_unused_parameters=False,
    )
    
    train_gflownet(model, train_set, eval_set, training_args)
