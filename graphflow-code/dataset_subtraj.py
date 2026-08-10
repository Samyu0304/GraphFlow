from torch.utils.data import Dataset
import random
import torch
from transformers import PreTrainedTokenizer
from stark_qa import load_qa, load_skb
from liquid import Template
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


flow_prompt = Template('''###Information trajectory you have visited:
{{history}}

###Question
{{question}}

Please predict the reward of the Information trajectory to the question.
''')
# stop when score>0.5 or choose the last one

policy_prompt = Template('''
###Information trajectory you have visited:
{{history}}

###Question
{{question}}

###Candidate Information
{{candidate}}

Please predict the score of the candidate to help you find the answer to the question.
''')

## candidate should contain the current node, so if choose the current node as candidate then end

class GFlowNetPathStepDataset(Dataset):
    def __init__(self, paths, skb_dataset, tokenizer, max_length=1024, num_negatives=5, doc_cutoff=500):
        self.paths = paths
        self.skb_dataset = skb_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.doc_cutoff = doc_cutoff
        self.flow_prompt = flow_prompt
        self.policy_prompt = policy_prompt

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = self.paths[idx]
        question = item["question"]
        path = item["path"]
        K = len(path) - 1

        if K == 0:
            raise ValueError("Path too short to sample a step.")

        # 随机采样一个 step
        i = random.randint(0, K - 1)
        current_node = path[i]
        true_next = path[i + 1]
        if i == 0:
            reward = 2
        elif i == K-1:
            reward = 0
        else:
            reward = 1

        # 构造 history 文本：state_i
        history_text = ''
        for j in range(i + 1):
            history_text += self.skb_dataset.get_doc_info(path[j])[:self.doc_cutoff] + "\n"

        state_text = self.flow_prompt.render(question=question, history=history_text.strip())
        state_inputs = self.tokenizer(state_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        # 构造 next state 文本：state_{i+1}
        next_history_text = ''
        for j in range(i + 2):  # 包含 i+1
            next_history_text += self.skb_dataset.get_doc_info(path[j])[:self.doc_cutoff] + "\n"

        next_state_text = self.flow_prompt.render(question=question, history=next_history_text.strip())
        next_state_inputs = self.tokenizer(next_state_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        # 构造 neighbor candidate
        neighbors = self.skb_dataset.get_neighbor_nodes(current_node)

        if not neighbors:
            candidate_nodes = [true_next] + [None for _ in range(self.num_negatives)]
            correct_idx = 0
        else:
            negative_candidates = [n for n in neighbors if n != true_next]
            negative_samples = random.sample(negative_candidates, min(self.num_negatives - 1, len(negative_candidates)))
            while len(negative_samples) < self.num_negatives - 1:
                negative_samples.append(None)

            if true_next != current_node:
                candidate_nodes = negative_samples + [current_node] + [true_next]
            else:
                candidate_nodes = negative_samples + [true_next] + [None]

            random.shuffle(candidate_nodes)
            correct_idx = candidate_nodes.index(true_next)

        neighbor_texts = [
            self.policy_prompt.render(question=question, history=history_text.strip(), candidate=self.skb_dataset.get_doc_info(n)[:self.doc_cutoff]) if n is not None else ""
            for n in candidate_nodes
        ]
        neighbor_inputs = self.tokenizer(neighbor_texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        return_data = {
            "state_input_ids": state_inputs["input_ids"].squeeze(0),  # [L]
            "state_attention_mask": state_inputs["attention_mask"].squeeze(0),
            "next_state_input_ids": next_state_inputs["input_ids"].squeeze(0),  # [L]
            "next_state_attention_mask": next_state_inputs["attention_mask"].squeeze(0),
            "neighbor_input_ids": neighbor_inputs["input_ids"],       # [N, L]
            "neighbor_attention_mask": neighbor_inputs["attention_mask"],
            "actions": torch.tensor([correct_idx], dtype=torch.long),
            "mask": torch.tensor([1.0]),  # always 1
            "reward": torch.tensor([reward]),
        }

        # # Debug print to check the shapes of the tensors
        # for key, value in return_data.items():
        #     print(f"{key}: {value.size()}")
        
        return return_data

def data_collator_fn(batch):
    # 用来存储批次数据的字典
    collated_data = {
        "state_input_ids": [],
        "state_attention_mask": [],
        "next_state_input_ids": [],
        "next_state_attention_mask": [],
        "neighbor_input_ids": [],
        "neighbor_attention_mask": [],
        "actions": [],
        "mask": [],
        "reward": []
    }

    # 遍历批次中的每个数据样本
    for sample in batch:
        collated_data["state_input_ids"].append(sample["state_input_ids"])
        collated_data["state_attention_mask"].append(sample["state_attention_mask"])
        collated_data["next_state_input_ids"].append(sample["next_state_input_ids"])
        collated_data["next_state_attention_mask"].append(sample["next_state_attention_mask"])
        collated_data["neighbor_input_ids"].append(sample["neighbor_input_ids"])
        collated_data["neighbor_attention_mask"].append(sample["neighbor_attention_mask"])
        collated_data["actions"].append(sample["actions"])
        collated_data["mask"].append(sample["mask"])
        collated_data["reward"].append(sample["reward"])

    # 将每个字段的列表转换为批次
    collated_data["state_input_ids"] = pad_sequence(collated_data["state_input_ids"], batch_first=True, padding_value=0)
    collated_data["state_attention_mask"] = pad_sequence(collated_data["state_attention_mask"], batch_first=True, padding_value=0)
    collated_data["next_state_input_ids"] = pad_sequence(collated_data["next_state_input_ids"], batch_first=True, padding_value=0)
    collated_data["next_state_attention_mask"] = pad_sequence(collated_data["next_state_attention_mask"], batch_first=True, padding_value=0)
    
    # 对邻居输入进行拼接，因为它是一个二维张量（N, L）
    collated_data["neighbor_input_ids"] = torch.stack(collated_data["neighbor_input_ids"], dim=0)
    collated_data["neighbor_attention_mask"] = torch.stack(collated_data["neighbor_attention_mask"], dim=0)
    
    # 处理其他字段，确保它们变为正确的形状
    collated_data["actions"] = torch.stack(collated_data["actions"], dim=0)
    collated_data["mask"] = torch.stack(collated_data["mask"], dim=0)
    collated_data["reward"] = torch.stack(collated_data["reward"], dim=0)

    return collated_data
