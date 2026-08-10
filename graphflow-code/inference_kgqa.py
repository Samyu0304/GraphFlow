import os
import tqdm
import json
import torch
import argparse
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
# from trl.trainer import reward_trainer
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import compute_accuracy, RewardDataCollatorWithPadding
from trl.data_utils import maybe_apply_chat_template
from accelerate import PartialState
from transformers.utils import is_peft_available
from peft import PeftModel, get_peft_model
import inspect
import warnings
from dataclasses import FrozenInstanceError, replace
# from typing import Any, Callable, Optional, Union
from liquid import Template
from liquid.template import BoundTemplate
from stark_qa import load_qa, load_skb
from random import sample
import json
import heapq
from collections import defaultdict
import numpy as np
import argparse
import re
import networkx as nx
from model import GFlowLlama

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="RoG-cwq", choices=["RoG-cwq", "webqsp"])
parser.add_argument("--best_model_path", type=str, default="/fs-computility/ai4sData/yujunchi/yujunchi/graphrag/gf_ckpt_safe/prime/32_16_1e-05_10_1_2_1024_0.25_4/best_model")

args = parser.parse_args()
dataset_repo_map = {
    "RoG-cwq": os.path.join("rmanluo", "RoG-cwq"),
    "webqsp": os.path.join("ml1996", "webqsp"),
}
test_set = load_dataset(
    dataset_repo_map[args.dataset_name],
    split="test",
    cache_dir="/fs-computility/ai4sData/yujunchi/yujunchi/graphrag/kgqa_dataset",
)


def build_knowledge_graph(triples, directed=False):
    """
    Build a NetworkX graph from a list of [head, relation, tail] triples.
    If directed=True, builds a DiGraph.
    Returns the graph obj.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    for head, relation, tail in triples:
        # Add nodes (optional: you can attach type or other metadata)
        G.add_node(head)
        G.add_node(tail)
        # Add edge with 'relation' as label attribute
        G.add_edge(head, tail, relation=relation)
    return G


pretrained_model_path = '/fs-computility/ai4sData/yujunchi/yujunchi/Meta-Llama-3-8B-Instruct'
dataset_name = args.dataset_name
peft_model_path = f"{args.best_model_path}/lora_adapter"
flow_head_path = f"{args.best_model_path}/flow_head.pt"
policy_head_path = f"{args.best_model_path}/policy_head.pt"
data_root = '/fs-computility/ai4sData/yujunchi/yujunchi/stark_data'
parts = args.best_model_path.split('/')
dataset_idx = parts.index('gf_ckpt_safe')
source_dataset = parts[dataset_idx + 1]
save_path = f"./graphrag_results/graphflow-source-{source_dataset}-target-{dataset_name}-LLaMA-results.json"

window_size = 3 # get latest 2 history
max_retrieve_steps = 5
max_length = 1024
doc_cutoff = 400
batch_size = 50
subgraph_dir = f'/fs-computility/ai4sData/yujunchi/yujunchi/G-Retriever-main/dataset/stark/{dataset_name}/cached_desc'
shortest_path_dir = f'/fs-computility/ai4sData/yujunchi/yujunchi/G-Retriever-main/dataset/stark/{dataset_name}/cached_paths'


def node_id_from_desc(file_id, subgraph_dir):
    file_path = f'{subgraph_dir}/{file_id}.txt'
    with open(file_path, 'r') as file:
        data = file.read()
    raw_node_list = re.findall(r"node_id,node_attr\n(.+?)\nsrc,edge_attr,dst", data, re.S)[0]
    raw_node_list = re.split('\n', raw_node_list)

    node_list = []

    for i in raw_node_list:
        if i != '':
            node_id = i.split(',')[0]
            node_list.append(int(node_id))
        
    return node_list

def edge_from_desc(file_id, subgraph_dir):
    file_path = f'{subgraph_dir}/{file_id}.txt'
    with open(file_path, 'r') as file:
        data = file.read()

    pattern = r'src,edge_attr,dst\n(.*)'
    match = re.search(pattern, data, re.DOTALL)
    results = match.group(1)
    results = results.split('\n')

    edge_list = []
    node_list = []
    for result in results:
        if result != '':
            result_list = result.split(',')
            source_node, target_node = result_list[0], result_list[2]
            edge_list.append((int(source_node), int(target_node)))
            node_list.append(int(source_node))
            node_list.append(int(target_node))

    node_list = list(set(node_list))

    return node_list, edge_list


from peft import PeftModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reward_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        )

reward_model = PeftModel.from_pretrained(reward_model, peft_model_path)
reward_model = reward_model.merge_and_unload()
backbone_device = next(reward_model.parameters()).device
model = GFlowLlama(llm_backbone=reward_model, use_lora=False)
flow_state_dict = torch.load(flow_head_path, map_location=backbone_device)
policy_state_dict = torch.load(policy_head_path, map_location=backbone_device)
model.flow_head.load_state_dict(flow_state_dict)
model.policy_head.load_state_dict(policy_state_dict)
model.flow_head.to(backbone_device)
model.policy_head.to(backbone_device)

print("Backbone:", next(model.backbone.parameters()).device)
print("Flow Head:", next(model.flow_head.parameters()).device)
print("Policy Head:", next(model.policy_head.parameters()).device)


reward_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
reward_tokenizer.padding_side = 'left'
# self.reward_tokenizer.padding_side = 'right'
reward_tokenizer.truncation_side = 'left'
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    # self.reward_tokenizer.pad_token = "<|end_of_text|>" # 128001
if reward_model.config.pad_token_id is None:
    # self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
    reward_model.config.pad_token_id = -1

policy_prompt = Template('''
###Information trajectory you have visited:
{{history}}

###Question
{{question}}

###Candidate Information
{{candidate}}

Please predict the score of the candidate to help you find the answer to the question.
''')

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def evaluate_metrics(source_list, target_list):
    """
    Calculate Hit@K, Recall@20, and Mean Reciprocal Rank (MRR) in percentage.
    
    Args:
        source_list (list): The list of recommended items.
        target_list (list): The list of ground-truth items.
    
    Returns:
        dict: A dictionary containing hit@1, hit@3, hit@5, hit@20, recall@20, and MRR values in percentage.
    """
    ks = [1, 3, 5, 20]
    hits_at_k = {f"hit@{k}": 0 for k in ks}
    recall_at_20 = 0
    mrr = 0
    
    for k in ks:
        top_k = source_list[:k]
        hits_at_k[f"hit@{k}"] = (1 if any(item in target_list for item in top_k) else 0) * 100
    
    top_20 = source_list[:20]
    recall_at_20 = (sum(1 for item in target_list if item in top_20) / len(target_list) * 100) if target_list else 0
    
    for idx, item in enumerate(source_list):
        if item in target_list:
            mrr = (1 / (idx + 1)) * 100
            break
    
    results = hits_at_k
    results["recall@20"] = recall_at_20
    results["mrr"] = mrr
    
    return results

# data = load_json(similarity_path)["qa"]
## get data split
skb_dataset = load_skb(dataset_name, root = data_root, download_processed=True)
qa_dataset = load_qa(dataset_name, root = data_root)

# get index split
dataset_splits = qa_dataset.get_idx_split()
test_indices = dataset_splits['test']
small_test_indice = dataset_splits['test-0.1']

og_result_dict = defaultdict(list)
pred_result_dict = defaultdict(list)
rerank_pred_result_dict = defaultdict(list)

for index in tqdm.tqdm(small_test_indice): # small_test_indice
    index = int(index)

    # get node and edge lists of pcst graphs
    node_list, edge_list = edge_from_desc(index, subgraph_dir)
    
    if len(node_list) > 200:
        continue
    edge_list = edge_list.T.tolist()
    question, _, answer_ids, _ = qa_dataset[index]
    #print(f'Answers: {answer_ids}')
    # merge path in the pcst graph

    # path json file
    fname = f'{shortest_path_dir}/{index}.json'
    if not os.path.exists(fname):
        continue

    with open(fname, 'r') as f:
        item = json.load(f)
    
    if item['path'] == []:
        print(f'Ignore file {fname}')
        continue

    predicted_ids = []
    all_predicted_paths = []

    for one_path_dict in tqdm.tqdm(item['path']):
        if one_path_dict['path'] == []:
            continue

        new_nodes = one_path_dict['path']
        if len(new_nodes) == 1:
            new_edges = edge_list
            new_nodes = node_list + new_nodes
        else:
            new_edges = [(new_nodes[i], new_nodes[i+1]) for i in range(len(new_nodes)-1)]
            new_edges = edge_list + new_edges
            new_nodes = node_list + new_nodes
        
        # construct nx graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(new_nodes)
        nx_graph.add_edges_from(new_edges)

        # get an initial state from new_nodes

        input_texts = [
            policy_prompt.render(question=question, history='', candidate=skb_dataset.get_doc_info(n)[:doc_cutoff]) if n is not None else ""
            for n in new_nodes
        ]
        inputs = reward_tokenizer(input_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        
        # inference 
        input_size = inputs["input_ids"].size()[0]
        num_batch = int(input_size/(batch_size+0.1)) + 1

        rewards = []
        for i in range(num_batch):
            batch_st = int(i * batch_size)
            batch_end = min(int((i+1)*batch_size), input_size)
            cur_input_ids=inputs["input_ids"][batch_st: batch_end].to(reward_model.device)
            cur_attention_mask=inputs["attention_mask"][batch_st: batch_end].to(reward_model.device)

            cur_rewards = model.predict_policy_scores(cur_input_ids,
                                cur_attention_mask)
            #print(rewards)
            # print(cur_rewards)
            cur_rewards = cur_rewards.tolist()
            rewards.extend(cur_rewards)
        
        init_node_candidates_id = list(map(rewards.index, heapq.nlargest(1, rewards)))
        initial_state_id = new_nodes[init_node_candidates_id[0]]

        # store history doc and pop recent doc
        history_doc_list = []
        cur_node_id = initial_state_id
        previous_node_id = -100 # avoid looking back to the previous nodes
        retrieval_steps = 1
        retrieve_path = [cur_node_id]


        while retrieval_steps <= max_retrieve_steps:

            print(f'retrieve {retrieval_steps} steps staring from Node {initial_state_id}. Now reach Node {cur_node_id}')

            cur_doc_info = skb_dataset.get_doc_info(cur_node_id)
            cur_doc_info = cur_doc_info[:doc_cutoff] + "\n" # some documents are very long
            neighborhood_node_ids = list(nx_graph.neighbors(cur_node_id))

            # check if the current node is isolated
            # print(f'Steps: {retrieval_steps} staring from {initial_state_id}')

            if neighborhood_node_ids == []:
                # predicted_ids.append(cur_node_id)
                retrieval_steps = 10000
            elif len(neighborhood_node_ids) >=400:
                retrieval_steps = 1000
                retrieve_path.append(initial_state_id)

            else:
                # if not, continue sampling
                # using window to filter out most up to date history
                # update history
                history_doc_list.append(cur_doc_info)

                filter_hsitory_doc_list = history_doc_list[len(history_doc_list)-window_size:]
                history_doc = '' 
                for i in filter_hsitory_doc_list:
                    history_doc += i
                
                neighborhood_node_ids.append(cur_node_id)
                input_texts = [
                    policy_prompt.render(question=question, history=history_doc.strip(), candidate=skb_dataset.get_doc_info(n)[:doc_cutoff]) if n is not None else ""
                    for n in neighborhood_node_ids
                ]
                inputs = reward_tokenizer(input_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            
                input_size = inputs["input_ids"].size()[0]
                # print("all input size", inputs["input_ids"].size())
                num_batch = int(input_size/(batch_size+0.1)) + 1

                rewards = []
                for i in range(num_batch):
                    batch_st = int(i * batch_size)
                    batch_end = min(int((i+1)*batch_size), input_size)
                    cur_input_ids=inputs["input_ids"][batch_st: batch_end].to(reward_model.device)
                    cur_attention_mask=inputs["attention_mask"][batch_st: batch_end].to(reward_model.device)
                    # print(i, cur_input_ids.size())

                    # cur_rewards = reward_model(cur_input_ids,
                    #                     cur_attention_mask,
                    #                     return_dict=True,)["logits"]
                    #print(rewards)
                    cur_rewards = model.predict_policy_scores(cur_input_ids,
                                                              cur_attention_mask)
                    cur_rewards = cur_rewards.tolist()
                    rewards.extend(cur_rewards)

                # next_node_id = neighborhood_node_ids[rewards.index(max(rewards))]
                two_next_node_candidates_id = list(map(rewards.index, heapq.nlargest(2, rewards)))
                #print(f'Next step arrives at {next_node_id}')
                top_1_candidate, top_2_candidate = neighborhood_node_ids[two_next_node_candidates_id[0]], neighborhood_node_ids[two_next_node_candidates_id[1]]
                
                # print(f'top1 candidates {top_1_candidate}, top 2 candidates {top_2_candidate}')
                if top_1_candidate != previous_node_id:
                    next_node_id = top_1_candidate
                else:
                    next_node_id = top_2_candidate

                if next_node_id != cur_node_id:
                    retrieve_path.append(next_node_id)
                    # iterate the retrieval_steps
                    retrieval_steps += 1
                    # update current node
                    previous_node_id = cur_node_id
                    cur_node_id = next_node_id

                else:
                    # predicted_ids.append(next_node_id)
                    retrieval_steps = 1000
            
        # use reward to rerank the outcome
        predicted_ids.append(retrieve_path[-1])
        # print(predicted_ids)
        all_predicted_paths.append(retrieve_path)
        # save to results
        
    
    #without rerank
    #print(predicted_ids, answer_ids)
    pred_metric_dict = evaluate_metrics(predicted_ids, answer_ids)
    
    
    # rerank the outputs
    input_texts = [
        policy_prompt.render(question=question, history='', candidate=skb_dataset.get_doc_info(n)[:doc_cutoff]) if n is not None else ""
        for n in predicted_ids
    ]
    inputs = reward_tokenizer(input_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    
    final_rewards = model.predict_policy_scores(inputs["input_ids"].to(reward_model.device), inputs["attention_mask"].to(reward_model.device))
    
    final_rewards = final_rewards.tolist()

    sorted_indices = sorted(range(len(final_rewards)), key=lambda i: final_rewards[i], reverse=True)
    sorted_predicted_ids = [predicted_ids[i] for i in sorted_indices]
    
    #print(sorted_predicted_ids, answer_ids)
    rerank_pred_metric_dict = evaluate_metrics(sorted_predicted_ids, answer_ids)
    
    
    # for key, value in og_metric_dict.items():
    #     og_result_dict[key].append(value)
    
    
    for key, value in pred_metric_dict.items():
        pred_result_dict[key].append(value)

    # print('------no rerank------')
    # print(f"Dataset: {dataset_name}")
    # for key, val in pred_result_dict.items():
    #     print(key, np.mean(val))
        
    for key, value in rerank_pred_metric_dict.items():
        rerank_pred_result_dict[key].append(value)

    # print('------rerank------')
    # for key, val in rerank_pred_result_dict.items():
    #     print(key, np.mean(val))
        
    
    # ---- 保存结果到 JSON 文件 ----
    result_entry = {
        "dataset": dataset_name,
        "index": index,
        "answer_ids": answer_ids,
        "no_rerank": {
            "metrics": {key: np.mean(val) for key, val in pred_result_dict.items()},
            "predicted_ids": predicted_ids
        },
        "rerank": {
            "metrics": {key: np.mean(val) for key, val in rerank_pred_result_dict.items()},
            "sorted_predicted_ids": sorted_predicted_ids
        }
    }

    # 选择保存路径

    # 如果文件存在就追加，否则新建
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result_entry)

    # 使用 indent=2 让 JSON 可读性更好
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)



                








