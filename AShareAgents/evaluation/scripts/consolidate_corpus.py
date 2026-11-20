"""Validate collected originals and publish one deduplicated corpus manifest."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, ROOT, CORPUS, read_jsonl, write_csv, write_jsonl, write_json, write_text,
)


def main():
    folder = CORPUS / 'manifests'
    company = read_jsonl(folder / 'company_documents.jsonl')
    policy = read_jsonl(folder / 'policy_documents.jsonl')
    rows, duplicates, seen = [], [], set()
    for row in [*company, *policy]:
        path = ROOT / row['raw_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']:
            raise ValueError('Original hash mismatch: ' + row['document_id'])
        if row['file_sha256'] in seen:
            duplicates.append(row['document_id'])
            continue
        seen.add(row['file_sha256'])
        rows.append(row)
    write_jsonl(folder / 'corpus_manifest.jsonl', rows)
    write_csv(folder / 'corpus_manifest.csv', rows)
    write_csv(folder / 'company_document_manifest.csv', company)
    write_csv(folder / 'policy_document_manifest.csv', policy)
    summary = {'documents': len(rows), 'companies': len({r['ticker'] for r in company}),
               'company_documents': len(company), 'policy_documents': len(policy),
               'company_industries': dict(Counter(r['industry'] for r in company)),
               'types': dict(Counter(r['document_type'] for r in rows)),
               'companies_distribution': dict(Counter(r.get('company_name') or '政策' for r in rows)),
               'date_status': dict(Counter(r['date_status'] for r in rows)),
               'unique_sha256': len(seen), 'duplicate_documents_excluded': duplicates,
               'raw_bytes': sum((ROOT / r['raw_path']).stat().st_size for r in rows)}
    write_json(folder / 'corpus_summary.json', summary)
    text = '# 真实语料收集报告\n\n' + '```json\n' + json.dumps(summary,ensure_ascii=False,indent=2) + '\n```\n\n'
    text += ('公司资料来自巨潮资讯官方披露记录，下载PDF与标题、股票代码、页数和SHA256核对；'
             '披露日来源announcementTime（Asia/Shanghai），没有从PDF文件名/URL猜测。'
             '政策原文来自官方发布机关或正式政府转载，日期依据实际页面发布标签，原始HTML及日期证据另存。'
             'date_status=verified表示机器核对官方来源，不能等同人工Golden标注verified。\n\n'
             '同一PDF/HTML的Markdown、正文txt、日期证据页不计为独立语料。101份原件按哈希去重。'
             '政策无原始PDF链接时保存原始HTML，逻辑页1不冒充PDF物理页。\n\n'
             '缺口：资料分布以可获取的定期报告为主，尚未每家公司都有2～3份IR/临时公告；'
             '公司采集期间巨潮部分分页请求403，保留原始响应元数据与失败日志，未宣称全量公告历史覆盖。'
             '两份2023政策保留，因为其规制对象及生效覆盖2024以后；政策失效/修订日期未知时为空。'
             '后续人工复核日期冲突（网站迁移元标签）、文档标题及政策附件完整性。\n\n'
             '复现：`python -B AShareAgents/evaluation/scripts/collect_company_documents.py --help`、'
             '`python -B AShareAgents/evaluation/scripts/collect_policy_documents.py --help`、'
             '`python -B AShareAgents/evaluation/scripts/consolidate_corpus.py`。具体选源与限制见manifests/company_validation.json、policy_collection_notes.md和各下载日志。\n')
    write_text(EVALUATION_DIR / 'reports/data_collection_report.md', text)
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
