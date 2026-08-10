from transformers import AutoModelForCausalLM
import torch.nn as nn
import torch

class GFlowLlama(nn.Module):
    def __init__(self, llm_backbone, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.backbone = llm_backbone

        if use_lora:
            from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
            # self.backbone = prepare_model_for_kbit_training(self.backbone)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            print("LoRA applied.")

        hidden_size = self.backbone.config.hidden_size
        # print(hidden_size)
        self.flow_head = nn.Sequential(nn.Linear(hidden_size, 1)).to(dtype=torch.bfloat16)
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU()).to(dtype=torch.bfloat16) # policy head output the scores for 

    
    def forward(self, state_input_ids, state_attention_mask, next_state_input_ids, next_state_attention_mask, neighbor_input_ids, neighbor_attention_mask, actions, mask, reward):
        B, N, L = neighbor_input_ids.size()  # N 是邻居候选数
        
        # print(f"state_input_ids: {state_input_ids.size()}")          # (B, L)
        # print(f"neighbor_input_ids: {neighbor_input_ids.size()}")    # (B, N, L)

        # ==== 1. State 推理 ====
        state_outputs = self.backbone(input_ids=state_input_ids, attention_mask=state_attention_mask, output_hidden_states=True)
        state_hidden = state_outputs.hidden_states[-1][:, 0, :]  # (B, hidden)
        flow = self.flow_head(state_hidden).squeeze(-1)          # (B,)

        # ==== 2. Next State 推理 ====
        #with torch.no_grad():
        next_state_outputs = self.backbone(input_ids=next_state_input_ids, attention_mask=next_state_attention_mask, output_hidden_states=True)
        next_state_hidden = next_state_outputs.hidden_states[-1][:, 0, :]  # (B, hidden)
        next_flow = self.flow_head(next_state_hidden).squeeze(-1)          # (B,)

        # ==== 3. Neighbor 推理 ====
        neighbor_input_ids_flat = neighbor_input_ids.view(B * N, L)
        neighbor_attention_mask_flat = neighbor_attention_mask.view(B * N, L)

        # with torch.no_grad():
        neighbor_outputs = self.backbone(input_ids=neighbor_input_ids_flat, attention_mask=neighbor_attention_mask_flat, output_hidden_states=True)
        neighbor_hidden = neighbor_outputs.hidden_states[-1][:, 0, :]  # (B*N, hidden)
        neighbor_hidden = neighbor_hidden.view(B, N, -1)               # (B, N, hidden)

        policy_scores = self.policy_head(neighbor_hidden).squeeze(-1)  # (B, N)
        policy_probs = torch.softmax(policy_scores, dim=-1)            # (B, N)

        return flow, next_flow, policy_probs
    
    @torch.no_grad()
    def predict_policy_scores(self, neighbor_input_ids, neighbor_attention_mask):
        """
        给定当前状态和邻居候选节点，输出 policy scores（未归一化 logits）
        输入：
            - state_input_ids: (B, L)
            - state_attention_mask: (B, L)
            - neighbor_input_ids: (B, N, L)
            - neighbor_attention_mask: (B, N, L)
        输出：
            - policy_scores: (B, N)
        """
        self.eval()  # 确保模型处于评估模式
        
        
        N, L = neighbor_input_ids.size()

        # # 1. 编码邻居
        # neighbor_input_ids_flat = neighbor_input_ids.view(B * N, L)
        # neighbor_attention_mask_flat = neighbor_attention_mask.view(B * N, L)

        neighbor_outputs = self.backbone(
            input_ids=neighbor_input_ids,
            attention_mask=neighbor_attention_mask,
            output_hidden_states=True
        )
        neighbor_hidden = neighbor_outputs.hidden_states[-1][:, 0, :]  # (B*N, hidden)
        neighbor_hidden = neighbor_hidden.view(N, -1)               # (B, N, hidden)

        # 2. 计算 policy scores
        policy_scores = self.policy_head(neighbor_hidden).squeeze(-1)  # (B, N)
        
        return policy_scores

    
def freeze_non_lora_and_heads(model):
    for name, param in model.named_parameters():
        # 保留 LoRA 参数
        if "lora_" in name:
            param.requires_grad = True
        # 保留自定义预测头参数
        elif "flow_head" in name or "policy_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    

if __name__ == '__main__':
    path = '/oss/ai4chem/chemshare/workspace/share/llms/Meta-Llama-3-8B-Instruct'
    model = GFlowLlama(path, use_lora=True)
    freeze_non_lora_and_heads(model)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


