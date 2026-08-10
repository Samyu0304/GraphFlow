import json
from collections import defaultdict

def compute_metrics(data):
    intervals = [(0, 5),(5, 10), (10, 15), (15, 20)]
    variants = ['rerank', 'no_rerank']
    
    # 初始化结果字典
    results = {
        variant: {
            (low, high): {
                "count": 0,
                "hit@1": 0,
                "hit@5": 0,
                "d-r@20": 0,
                "mrr": 0
            } for (low, high) in intervals
        } for variant in variants
    }

    for item in data:
        answer_ids = set(item["answer_ids"])
        ans_len = len(answer_ids)
        for (low, high) in intervals:
            if low <= ans_len < high:
                for variant in variants:
                    
                    if variant == 'rerank':
                        pred_key = 'sorted_predicted_ids'
                    
                    else:
                        pred_key = "predicted_ids"
                        
                    preds = item[variant][pred_key][:20]
                    preds_dedup = list(dict.fromkeys(preds))  # 去重保持顺序

                    bucket = results[variant][(low, high)]
                    bucket["count"] += 1

                    # hit@1
                    if preds and preds[0] in answer_ids:
                        bucket["hit@1"] += 1

                    # hit@5
                    if any(pid in answer_ids for pid in preds[:5]):
                        bucket["hit@5"] += 1

                    # De-duplicate Recall@20
                    match_count = sum(1 for pid in preds_dedup if pid in answer_ids)
                    recall = match_count / len(answer_ids) if answer_ids else 0
                    bucket["d-r@20"] += recall

                    # MRR
                    rr = 0
                    for rank, pid in enumerate(preds):
                        if pid in answer_ids:
                            rr = 1 / (rank + 1)
                            break
                    bucket["mrr"] += rr
                break  # 找到匹配区间就跳出

    # 计算平均值
    for variant in variants:
        for key, val in results[variant].items():
            count = val["count"]
            if count > 0:
                val["hit@1"] /= count
                val["hit@5"] /= count
                val["d-r@20"] /= count
                val["mrr"] /= count
            else:
                val["hit@1"] = val["hit@5"] = val["d-r@20"] = val["mrr"] = None

    return results


# 主程序入口
if __name__ == "__main__":
    with open("/fs-computility/ai4sData/yujunchi/yujunchi/graphrag/llm_inference_results/vallinaprm-source-amazon-target-amazon-LLaMA-results.json", "r") as f:
        data = json.load(f)

    metrics = compute_metrics(data)

    for variant in ['rerank', 'no_rerank']:
        print(f"\n=== {variant.upper()} Metrics ===")
        for interval, values in metrics[variant].items():
            print(f"Range {interval}:")
            print(f"  Count: {values['count']}")
            print(f"  Hit@1:   {values['hit@1']:.3f}" if values['hit@1'] is not None else "  Hit@1:   N/A")
            print(f"  Hit@5:   {values['hit@5']:.3f}" if values['hit@5'] is not None else "  Hit@5:   N/A")
            print(f"  D-R@20:  {values['d-r@20']:.3f}" if values['d-r@20'] is not None else "  D-R@20:  N/A")
            print(f"  MRR:     {values['mrr']:.3f}" if values['mrr'] is not None else "  MRR:     N/A")
            print()