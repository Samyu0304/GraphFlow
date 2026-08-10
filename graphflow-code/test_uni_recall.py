import json

def compute_metrics(results, k=20, rerank=True):
    total_recall = 0
    total_diversity = 0
    total_samples = 0

    for item in results:
        answer_ids = set(item["answer_ids"])
        if not answer_ids:
            continue  # 跳过无标注项

        if rerank:
            pred_ids = item["rerank"]["sorted_predicted_ids"][:k]
        else:
            pred_ids = item["no_rerank"]["predicted_ids"][:k]

        # Recall: 命中的 unique answer 数 / answer 总数
        unique_hits = set(pred_ids) & answer_ids
        dedup_recall = len(unique_hits) / len(answer_ids)

        # Diversity: top-k 预测中不重复的占比
        diversity = len(set(pred_ids)) / k

        total_recall += dedup_recall
        total_diversity += diversity
        total_samples += 1

    avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
    avg_diversity = total_diversity / total_samples if total_samples > 0 else 0.0
    return avg_recall, avg_diversity

# 加载 JSON 文件
with open("/fs-computility/ai4sData/yujunchi/yujunchi/graphrag/llm_inference_results/tog-prime-LLaMA-results.json", "r") as f:
    results = json.load(f)

# 计算 @20 的 recall 和 diversity
recall_rerank, diversity_rerank = compute_metrics(results, k=20, rerank=True)
recall_no_rerank, diversity_no_rerank = compute_metrics(results, k=20, rerank=False)

print(f"[Rerank]     Dedup Recall@20: {recall_rerank:.4f}, Diversity@20: {diversity_rerank:.4f}")
print(f"[No Rerank]  Dedup Recall@20: {recall_no_rerank:.4f}, Diversity@20: {diversity_no_rerank:.4f}")