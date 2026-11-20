"""Hand-checkable IR metrics. Grade 2 is positive; nDCG uses grades 0/1/2."""
import math


def ranking_metrics(ranking, qrels, ks=(5, 10, 20), positive_grade=2):
    ranking = list(dict.fromkeys(ranking))
    positives = {key for key, grade in qrels.items() if grade >= positive_grade}
    result = {'returned_count': len(ranking), 'relevant_count': len(positives)}
    for k in ks:
        chosen = ranking[:k]
        result[f'recall@{k}'] = (len(set(chosen) & positives) / len(positives)
                                 if positives else None)
        result[f'mrr@{k}'] = (next((1 / rank for rank, key in enumerate(chosen, 1)
                                    if key in positives), 0.0) if positives else None)
        gains = [2 ** max(0, min(2, int(qrels.get(key, 0)))) - 1 for key in chosen]
        dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
        ideal = sorted((2 ** max(0, min(2, int(x))) - 1 for x in qrels.values()), reverse=True)[:k]
        idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
        result[f'ndcg@{k}'] = dcg / idcg if idcg else None
    return result
