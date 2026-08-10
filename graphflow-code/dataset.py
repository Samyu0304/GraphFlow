from torch.utils.data import Dataset
import random
import torch
from transformers import PreTrainedTokenizer
from stark_qa import load_qa, load_skb
from liquid import Template
import torch.nn as nn


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

class GFlowNetPathDataset(Dataset):
    def __init__(self, paths, skb_dataset, tokenizer: PreTrainedTokenizer, max_length=4096, num_negatives=5, doc_cutoff=500):
        """
        paths: List of node ID paths, e.g. [[0, 1, 2, 3], [5, 6, 7, 8]]
        graph: Dict[node_id] -> Dict with keys: "text" and "neighbors" (List[node_id])
        tokenizer: Huggingface tokenizer
        """
        self.paths = paths # paths = [path_1, path_2, ... ,path_n]  path_n = {'question': question, 'path': path}
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
        path = self.paths[idx]

        question = path['question'] ## placeholder for questions
        path = path['path']
        # here path is the real path
        K = len(path) - 1

        step_node_texts = []  # shape: [K, N]
        action_indices = []  # shape: [K]
        node_inputs = []     # shape: [K+1, L]
        
        # concate the history information
        node_texts = [self.skb_dataset.get_doc_info(n)[:self.doc_cutoff] for n in path] # the node ids
        
        state_doc_list = []
        history_doc_list = []
        history_text = ''
        for i in range(len(node_texts)):
            history_text = history_text + node_texts[i] + '\n'
            history_doc_list.append(history_text)
            warp_state_text = self.flow_prompt.render(question=question, history=history_text)
            state_doc_list.append(warp_state_text)

        # if reach the last node, the policy network will take 
        for i in range(K + 1): # all node list
            current_node = path[i]
            
            if i == K:
                true_next = path[i] # the end state
            else:
                true_next = path[i + 1]
            neighbors = self.skb_dataset.get_neighbor_nodes(current_node)

            # 如果当前节点无邻居（孤立节点）
            if not neighbors:
                candidate_nodes = [true_next] + [None for i in range(self.num_negatives)]
                correct_idx = 0
                # step_node_texts.append([self.skb_dataset.get_doc_info(true_next)[:self.doc_cutoff]]) 
                # action_indices.append(correct_idx)
                # continue
            
            else:

                # 确保 true_next 在邻居中, 但是会导致bug, 比如
                # if true_next not in neighbors:
                #     neighbors.append(true_next)

                # 过滤掉 true_next，采样负样本
                # 这里的true next还是node id，转换完成后一起变成text information
                negative_candidates = [n for n in neighbors if n != true_next] 
                negative_samples = random.sample(negative_candidates, min(self.num_negatives-1, len(negative_candidates))) # N-1
                # negative_samples.append(current_node) # N + 1
                
                while len(negative_samples) < self.num_negatives - 1:
                    negative_samples.append(None) # N
                
                #print(negative_samples, current_node, true_next, current_node==true_next)
                if true_next != current_node:
                    candidate_nodes = negative_samples + [current_node] + [true_next] # N + 1
                else:
                    candidate_nodes = negative_samples + [true_next] + [None] # N + 1
                # if the true_next is not the current_node, we add current_node in the candidate nodes
                #print(candidate_nodes)
                
                random.shuffle(candidate_nodes)
                correct_idx = candidate_nodes.index(true_next)

            # print(f"number of candidate nodes: {len(candidate_nodes)}")

            step_node_texts.append([
                self.policy_prompt.render(question=question, history = history_doc_list[i], candidate = self.skb_dataset.get_doc_info(n)[:self.doc_cutoff]) if n is not None else "" for n in candidate_nodes
            ])
            action_indices.append(correct_idx)

        # encode state nodes in path

        
        state_inputs = self.tokenizer(state_doc_list, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        neighbor_inputs = [
            self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            for texts in step_node_texts
        ]

        neighbor_input_ids = torch.stack([n["input_ids"] for n in neighbor_inputs])  # [K, N, L]
        neighbor_attention_mask = torch.stack([n["attention_mask"] for n in neighbor_inputs])
        
        return_data = {
            "state_input_ids": state_inputs["input_ids"],                # [K+1, L]
            "state_attention_mask": state_inputs["attention_mask"],
            "neighbor_input_ids": neighbor_input_ids,                  # [K+1, N, L]
            "neighbor_attention_mask": neighbor_attention_mask,
            "actions": torch.tensor(action_indices, dtype=torch.long), # [K+1]
            "mask": torch.ones(len(path), dtype=torch.float),        # [K+1]
        }
        
        # print(return_data.keys())

        return return_data

def pad_tensor(t, target_len, dim=0, pad_value=0):
    pad_size = list(t.shape)
    pad_size[dim] = target_len - t.shape[dim]
    if pad_size[dim] <= 0:
        return t
    padding = torch.full(pad_size, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, padding], dim=dim)

def data_collator_fn(batch):
    batch_dict = {key: [item[key] for item in batch] for key in batch[0]}
    max_len = max(x.shape[0] for x in batch_dict["state_input_ids"])  # max(K+1)

    # padding for each field with shape [K+1, ...]
    for key in ["state_input_ids", "state_attention_mask", "actions", "mask"]:
        batch_dict[key] = [
            pad_tensor(x, max_len, dim=0, pad_value=0) for x in batch_dict[key]
        ]

    for key in ["neighbor_input_ids", "neighbor_attention_mask"]:
        batch_dict[key] = [
            pad_tensor(x, max_len, dim=0, pad_value=0) for x in batch_dict[key]
        ]

    return {
        "state_input_ids": torch.stack(batch_dict["state_input_ids"]),             # [B, K+1, L]
        "state_attention_mask": torch.stack(batch_dict["state_attention_mask"]),   # [B, K+1, L]
        "neighbor_input_ids": torch.stack(batch_dict["neighbor_input_ids"]),       # [B, K+1, N, L]
        "neighbor_attention_mask": torch.stack(batch_dict["neighbor_attention_mask"]),
        "actions": torch.stack(batch_dict["actions"]),                             # [B, K+1]
        "mask": torch.stack(batch_dict["mask"]),                                   # [B, K+1]
    }
# def gflownet_collate_fn(batch):
#     print(len(batch))
#     print(batch[0].keys())
#     max_K = max(sample["actions"].shape[0] for sample in batch)
#     num_candidates = batch[0]["neighbor_input_ids"].shape[1]
#     max_len = batch[0]["state_input_ids"].shape[1]

#     def pad_tensor(t, pad_dim, max_len):
#         pad_size = [0] * (2 * t.dim())
#         pad_size[2 * pad_dim + 1] = max_len - t.size(pad_dim)
#         return nn.functional.pad(t, pad=pad_size, value=0)

#     node_input_ids = torch.stack([
#         pad_tensor(sample["state_input_ids"], pad_dim=0, max_len=max_K + 1)
#         for sample in batch
#     ])
#     node_attention_mask = torch.stack([
#         pad_tensor(sample["state_attention_mask"], pad_dim=0, max_len=max_K + 1)
#         for sample in batch
#     ])

#     neighbor_input_ids = torch.stack([
#         pad_tensor(sample["neighbor_input_ids"], pad_dim=0, max_len=max_K)
#         for sample in batch
#     ])
#     neighbor_attention_mask = torch.stack([
#         pad_tensor(sample["neighbor_attention_mask"], pad_dim=0, max_len=max_K)
#         for sample in batch
#     ])

#     actions = torch.stack([
#         nn.functional.pad(sample["actions"], pad=(0, max_K - sample["actions"].shape[0]), value=-100)
#         for sample in batch
#     ])
#     # rewards = torch.stack([sample["reward"] for sample in batch])
#     masks = torch.stack([
#         nn.functional.pad(sample["mask"], pad=(0, max_K - sample["mask"].shape[0]), value=0)
#         for sample in batch
#     ])

#     return {
#         "state_input_ids": node_input_ids,
#         "state_attention_mask": node_attention_mask,
#         "neighbor_input_ids": neighbor_input_ids,
#         "neighbor_attention_mask": neighbor_attention_mask,
#         "actions": actions,
#         "mask": masks,
#     }