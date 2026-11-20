"""Render the final report from observed artifacts; never fabricate missing stages."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, RETRIEVAL_CONFIG, ROOT, CORPUS, read_jsonl, write_json, write_text,
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main():
    corpus=read(CORPUS/'manifests/corpus_summary.json')
    ingest=read(CORPUS/'manifests/ingestion_summary.json')
    quality=read(EVALUATION_DIR / 'reports/dataset_quality_report.json')
    retrieval=read(EVALUATION_DIR / 'results/retrieval_summary.json')
    analysis=read(EVALUATION_DIR / 'results/retrieval_analysis.json')
    generation=read(EVALUATION_DIR / 'results/generation_analysis.json')
    selected=read(EVALUATION_DIR / 'results/selected_strategy.json')
    run=read(EVALUATION_DIR / 'results/run_summary.json')
    original_hash=hashlib.sha256((ROOT/'cache/rag/knowledge.db').read_bytes()).hexdigest()
    original_unchanged=original_hash=='87b97ed773a8e49ff0dc8f240eb8003e3a4bd614f9524da6d33fed47c77a4a7e'
    cloud_path=CORPUS/'knowledge_base/cloud_cache.sqlite'
    with sqlite3.connect('file:'+str(cloud_path)+'?mode=ro',uri=True) as db:
        cloud_attempts,cloud_reserved=db.execute('SELECT count(*),COALESCE(sum(estimated_cny),0) FROM paid_reservations').fetchone()
    ledger=CORPUS/'cloud_generation_budget.sqlite'
    annotation_cost=4.408048
    annotation_ledger=CORPUS/'cloud_annotation_budget.sqlite'
    if annotation_ledger.exists():
        with sqlite3.connect('file:'+str(annotation_ledger)+'?mode=ro',uri=True) as db:
            annotation_cost=db.execute('SELECT reserved_microcny FROM generation_budget').fetchone()[0]/1e6
    gen_cost=0
    if ledger.exists():
        with sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True) as db:
            gen_cost=db.execute('SELECT reserved_microcny FROM generation_budget').fetchone()[0]/1e6
    budget={'authorized_cny':50,'annotation_conservative_reserved_cny':annotation_cost,
            'embedding_rerank_conservative_reserved_cny':cloud_reserved,'embedding_rerank_attempts':cloud_attempts,
            'generation_usage_settled_plus_unsettled_reserved_cny':gen_cost,
            'combined_conservative_cost_cny':annotation_cost+cloud_reserved+gen_cost,
            'not_provider_invoice':True}
    write_json(EVALUATION_DIR / 'results/final_budget_summary.json',budget)
    testlog=CORPUS/'manifests/final_engineering_tests.log'
    log=testlog.read_text(encoding='utf-8',errors='replace') if testlog.exists() else ''
    test_result=next((line.strip('= \n\r') for line in reversed(log.splitlines()) if re.search(r'\d+ passed',line)),'尚未取得完整测试通过记录')
    report=f'''# RAG_EVALUATION_FINAL_REPORT

生成时间：{datetime.now(timezone.utc).isoformat()}。项目根目录：`{ROOT}`。代码基线：`d35d2f176549f2ee00534b0fa1166b0dd8e55801`，分支 `feat/ashare-llm-guardrails`。保留任务前未提交修改；未 commit、push、reset，未启动完整多Agent工作流。

## 1. 原有架构与本次改造

原有RAGService按company_official/policy_industry分流，DataPreparationModule建立父子块，IndexConstructionModule调用云端Embedding及SQLite，RetrievalOptimizationModule实现BM25、Dense、RRF和Reranker，GenerationIntegrationModule构建上下文。本次沿用这些算法和业务分类，未部署Milvus或大型本地模型。

必要改造：新增`pdf_parser.py`逐页解析与显式质量标记；`metadata.py`保留来源/日期/原件哈希/页码；`data_preparation.py`增加页级稳定父子ID；SQLite保留扩展字段并支持类型、报告期、发布日期、生效/失效日过滤；修复禁用重排器仍构建云客户端以及本地旧配置兼容。已有云适配器与原数据库保留。评测脚本独立编排，复用生产分词、BM25Okapi、余弦、RRF、父块提升及精排。

## 2. 真实文档收集

实收101份独立原件：10家公司76份PDF、25份政策（4PDF、21官方HTML），101个独立SHA256；公司覆盖9行业。公司资料20年报、19中报、28季报、9临时公告。2024/2025年度报告及对应2025/2026资料以实际披露时间为准。完整公司、文档及来源清单见`corpus_manifest.csv`、`company_document_manifest.csv`、`policy_document_manifest.csv`。

公司来源巨潮官方披露记录，日期依据归档announcementTime；政策来源发布机构或政府正式转载，保留原文、日期证据与正文，未用媒体评论替代。网站迁移元标签与正文日期冲突保留核验依据；未知修订/失效日为空。date_status=verified是机器核对官方元信息，并非人工Golden认证。8份前期候选原件未计入76份，未删除。未把PDF和派生txt/JSON重复计数。未达到每家公司2～3份IR/公告的理想类型配比。

## 3. 解析与独立SQLite入库

```json
{json.dumps(ingest,ensure_ascii=False,indent=2)}
```

101份具有解析产物，10,098个物理/逻辑页；23,133父块、100,846子块为冻结解析快照。21HTML使用逻辑页1。34空文本/疑似扫描页未做OCR；招商银行4份年报/中报共1,183页发生字体映射乱码，已隔离，题目改用清晰三季报。这4份只能认定为部分解析，不能将101份产物称作101份完整准确正文。9,744页带表格layout未核验标志；未做OCR、图形理解、财务表格语义重建或可靠章节层级识别。

独立库`data/rag_eval_v1/knowledge_base/knowledge_v1.sqlite`；逐文档SHA256+parse_key+实际记录数判断重复导入，模型/维度配置冻结。100,846条向量的完成状态以上述真实计数为准，不以代码存在推定运行完成。根据子Chunk ID可定位parent_id、document_id、manifest原件、URL、哈希与页码；原始页保留。

101份扩充语料位于独立评测库，生产Agent默认库未切换；原业务库仍为原先42子块/13父块的单文档基线。后续若需要上线扩充语料，应先完成人工质量复核，再单独评估配置切换。

曾遇百炼403 `AccessDenied.Unpurchased`，停止批量请求，用户确认控制台开通后从缓存恢复。失败日志与诊断留存，没有把失败当成有效向量。PyMuPDF加速安装曾被自动审批拒绝，最终使用已有pypdf，不绕过拒绝。原业务库SHA256现为`{original_hash}`；与基线一致：**{original_unchanged}**。

## 4. Golden Dataset

实际{quality.get('questions','尚未确认')}题，170可回答、20无答案；六类35/35/30/30/30/30，固定seed20261116划分38Dev/152Test。证据页、逐字引文、父子ID和时间一致性自动检查错误数{len(quality.get('errors',[]))}；结构性证据完整率{quality.get('evidence_completeness')}（引用存在且精确映射，不等于所有原子事实已语义核验）。全部190题pending_review，人工verified=0。

公司草稿发现并修正30处语义/数值问题；例如宁德分红报告期、恒瑞未经来源支持的金额、工商不良率、中芯现金流方向、隆基重列基数、紫金EBITDA误当经营现金流。另修正plain/layout列顺序引文差异，边界引文保留全部必要子块片段。修复有完整前后审计记录，不能据此替代人工核验。

三级相关性2/1/0，父子qrels分别输出TREC TSV及JSONL；0来自显式困难/未来负例。未标注块是未判断，计算时按0会产生不完整qrels偏差。两对相似题已标注保留理由，其中一对为同公司不同业务风险且跨Dev/Test，有内容相关性，不能声称严格语义族隔离。20无答案主要考察披露前边界，开放域无答案和真实冲突类型仍应扩充。

## 5. 四种检索策略实际对照

```json
{json.dumps(retrieval,ensure_ascii=False,indent=2)}
```

Dense20、BM2520、RRF常数60、Reranker候选30、最终生成Top5；诊断指标算到K20。BM25全量合格池，无chunk_id字典序任意截断；有界2池缓存仅复用同生产分词与BM25索引，回归验证排名/分数一致。原始候选、父子排名、分数、日期、过滤、延迟全部保存；断点严格匹配dataset/corpus/config/query指纹。父块在子块K截断前提升。无答案不强算Recall/MRR/nDCG。

生成配置选择：`{selected.get('strategy','尚未选择')}`，仅依据Dev父块nDCG@5，Test未调参。全部标签未人工核验，结果均为暂定。

原始corpus_hash只标识文档字节；新增knowledge_snapshot_hash把全部101文档parse_key、元数据/分块配置与Embedding身份纳入语义快照。当前760检索及190生成记录经过逐项一致性核对后补充快照标签，明确标注post_run_verified_frozen_snapshot、recorded_at_execution=false，未冒充执行时已记录。见snapshot_attestation.json。

'''
    ag=[r for r in analysis.get('aggregates',[]) if r['split']=='test' and r['grain']=='parent']
    report+='|策略|Test父块Recall@5|Recall@10|Recall@20|MRR@5|nDCG@5|\n|---|---:|---:|---:|---:|---:|\n'
    for r in ag:
        report+='|'+r['strategy']+'|'+ '|'.join(f'{r[k]:.4f}' if r.get(k)is not None else 'N/A' for k in ('recall@5','recall@10','recall@20','mrr@5','ndcg@5'))+'|\n'
    report+='\n完整父子、类别、Dev/Test与K指标及具体失败案例见`retrieval_evaluation_report.md`，不得只比较单一均值。\n\n'
    report+=f'''## 6. RAGAS生成质量实际结果

```json
{json.dumps(generation,ensure_ascii=False,indent=2)}
```

实际使用隔离RAGAS0.4.4 Collections ContextPrecision、ContextRecall、Faithfulness与云端deepseek-flash；固定Prompt/temperature0/输出上限2048。只执行所选配置，未对四组重复全量生成。生成提示词不含标准答案，判断阶段单独接收reference。失败评分保留缺失，不补0。无答案另以独立云评测员判断拒答正确率和无依据回答率，明确为暂定模型判断，不冒充人工确认结果。

首轮39个Faithfulness结构化输出因2048上限截断，使用同一模型/温度/评分Prompt只重试缺失指标，判断输出上限8192，实际78次HTTP；原回答与已有有效分数未改变，39项全部修复，170条三指标无缺失。分项修复配置保存在metric_repairs及metric_repair_summary.json；未悄悄换模型或把失败记0。完整结果再次执行相同生成命令，190条全部命中已完成快照、0次外部调用，见resume_validation.json。

RAGAS整体均值覆盖全部170条可回答题，同时单列35Dev/135Test；不能把整体值冒充纯Test泛化成绩。IR Recall以标注的关键块为单位，RAGAS Context Recall由模型判断参考事实覆盖，二者定义和不完整标注偏差不同，不能直接等同。

## 7. 关键失败与证据链问题

- 草稿真实存在财务口径错误，来源逐字匹配不足以证明答案语义正确；人工复核仍是必要环节。
- PDF字体映射、列顺序与非打印字体符号导致精确引文初次映射失败；修复只做可审计格式对齐，未做模糊数字替换。
- 多段/跨期问题需要多个关键证据；首个命中不代表全部召回，父子粒度和候选预算不能混用。
- 精排只能改善已有候选，无法补回Dense/BM25候选池外证据。低分案例应结合真实Top结果与原始页判断，详见专门报告，不从平均分推断机制。
- 例如ragv1_000333-01询问2024营业总收入，精排Top1却来自2025年报；两份报告均早于2026-11-16，因此这是报告期匹配失败，不能误称发布日期过滤失效。ragv1_000333-10检索到了2025现金流附注细项，却缺少用于2024/2025经营现金流比较的两年总额关键行，体现语义相似和完整证据覆盖的差别。
- 如仍有生成或判断错误，逐题metric_errors、原始回答、完整上下文与时间记录保存在generated_answers.jsonl。

## 8. 系统不足与实际边界

真实文字质量及财务表格结构仍不完整；原件覆盖不等于全文可用。政策4项有明确生效日，其余未知，不能以披露日代替生效日。版本别名未锁定供应商内部模型修订，未来复跑不保证数值逐位一致。云缓存包含模型/维度/主机/文本及有序候选，但同主机端点路径变更需主动清缓存或升级版本。日期过滤具备工程单测与本次实际日志验证，尚无全部历史时点/公司组合覆盖。

as_of_date按披露日期日粒度包含当日资料，不能证明开盘或收盘前已公开。1个子块长度205字符，其余不超过200；全部父块不超过800，实际统计见chunk_size_quality.json。无答案集主要是时间边界，不足以代表所有开放域或真实冲突拒答场景。

## 9. 后续优化建议

先人工核验190题、全部原子事实、qrels完整性和无答案闭集范围，再冻结v1.0。对乱码PDF用可验证替代官方原件或轻量受控OCR修复；重点复核表头、合并单元格和单位。补充各公司IR/重大公告、扫描和表格困难样本；增加非时间型无答案与真正冲突证据。重新标注新增等价相关块，实施语义族隔离划分，再于Dev尝试候选预算与分块参数，保持Test冻结。后续Agent场景可复用本次结构化引用链，另行验证工具选择、分析日期和报告事实，不在本任务改造Agent流程。

## 10. 实际命令、测试与关键配置

实际工程主环境结果：**{test_result}**。隔离RAGAS真实API的离线HTTP fixture测试见ragas_environment_report.md；fixture不属于正式语料或真实评分。Windows受限socket环境曾使完整测试停滞，改为仍禁外部网络的实际进程运行后取得完整日志。

隔离RAGAS环境最终38项通过，pip check无损坏依赖。基础Conda：Python3.11.17/pypdf7.1.0/LangChain-core1.7.0/LangGraph1.2.17/OpenAI3.37.0，未安装RAGAS；独立venv：RAGAS0.4.4/LangChain-core1.8.1/OpenAI2.54.0，LangGraph仍1.2.17。完整版本记录见final_base_environment.json、final_ragas_environment.json。下列python均指C:/Users/QWQ/.conda/envs/ashareagents/python.exe。

实际命令（按阶段运行或断点恢复，非每条均无错误；失败日志保留）：

```powershell
python -B AShareAgents/evaluation/scripts/collect_company_documents.py --help
python -B AShareAgents/evaluation/scripts/collect_policy_documents.py --help
python -B AShareAgents/evaluation/scripts/consolidate_corpus.py
python -B AShareAgents/evaluation/scripts/ingest_corpus.py --workers 2
python -B AShareAgents/evaluation/scripts/embed_staged_corpus.py --allow-paid --cloud-budget-cny 20
python -B AShareAgents/evaluation/scripts/embed_staged_corpus.py --allow-paid --cloud-budget-cny 15
python -B AShareAgents/evaluation/scripts/build_golden_dataset.py
python -B AShareAgents/evaluation/scripts/run_retrieval_eval.py --strategies BM25
python -B AShareAgents/evaluation/scripts/run_retrieval_eval.py --allow-paid --cloud-budget-cny 17
./data/rag_eval_v1/runtime_ragas/venv/Scripts/python.exe -B AShareAgents/evaluation/scripts/run_generation_eval.py --strategy '{selected.get('strategy','待选择')}' --model deepseek-flash --judge-model deepseek-flash --allow-paid --max-queries 190 --max-calls 2200 --budget-cny 28 --max-output-tokens 2048 --workers 8
./data/rag_eval_v1/runtime_ragas/venv/Scripts/python.exe -B AShareAgents/evaluation/scripts/repair_generation_metrics.py --allow-paid --budget-cny 28 --max-calls 600 --judge-output-limit 8192 --workers 8
python -B AShareAgents/evaluation/scripts/attest_snapshot.py --include-generation
python -B AShareAgents/evaluation/scripts/summarize_evaluation.py
python -B -u -m pytest tests -vv -o faulthandler_timeout=60
```

采集的实际参数与失败请求另见两个collection脚本、manifest元数据、下载日志。首次embedding20元分项上限后收紧为15，后续检索17，生成28，注释5；后两项与注释上界总计50。实际费用监控如下，均为保守预留/usage峰值估价，**不是供应商账单**：

```json
{json.dumps(budget,ensure_ascii=False,indent=2)}
```

成功生成/判断依据合法usage原子结算预留，只结算一次；失败/崩溃/缺usage保持上界。向量/重排保留每attempt字节上界预留。`cost_budget.json`有价格来源，真实账单需用户在控制台核对。独立RAGASvenv未升级主Conda；`.env`未修改，未输出密钥。

## 11. 尚未完成与状态区分

- **已完成并实际运行**：真实原件采集、页级解析、190候选题、精确映射与自动校验；向量化、检索、生成的完成数量必须以上述真实计数为准。
- **数据已准备但未人工核验**：190题答案、原子事实、三级qrels、20无答案判定；人工verified仍0。
- **代码具备但覆盖不足**：扫描OCR未实现；自动章节/复杂表格语义结构未实现；政策失效与修订日期缺乏完整官方证据。
- **外部限制历史记录**：巨潮部分分页403、百炼付费访问403；均保留日志，百炼经用户开通后续跑。若任一实际计数仍小于101/190/四组全量，需认定该阶段尚未完成，不能用脚本存在替代成功。

## 12. 核心文件路径

- `{ROOT/'AShareAgents/rag/pdf_parser.py'}`、`metadata.py`、`data_preparation.py`、`service.py`；`AShareAgents/storage/sqlite_knowledge_store.py`。
- `{CORPUS/'manifests/corpus_manifest.csv'}`；company/policy_document_manifest.csv；company_seed_rescue_audit.json、company_seed_layout_rescue_audit.json、parsing_quality_summary.json。
- `{CORPUS/'knowledge_base/knowledge_v1.sqlite'}`；cloud_cache.sqlite；cloud_generation_budget.sqlite。
- `{EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl'}`；dev.jsonl、test.jsonl、qrels_parent.tsv、qrels_child.tsv。
- `{EVALUATION_DIR / 'annotation/human_review_queue.csv'}`、annotation_guidelines.md。
- `{RETRIEVAL_CONFIG}`、cost_budget.json、requirements-ragas.txt。
- `{EVALUATION_DIR / 'results'}`：retrieval_runs.jsonl、retrieval_raw_candidates.jsonl、retrieval_rankings.csv、retrieval_metrics.csv、retrieval_analysis.json、generated_answers.jsonl、ragas_metrics.csv、generation_analysis.json、final_budget_summary.json。
- `{EVALUATION_DIR / 'reports'}`：data_collection_report.md、ingestion_report.md、dataset_quality_report.md、retrieval_evaluation_report.md、ragas_evaluation_report.md、ragas_environment_report.md。
- `{CORPUS/'manifests/final_engineering_tests.log'}`、final_isolated_ragas_tests.log；cloud_embedding_run/resume.log及检索/生成运行日志。所有大原件、解析正文、数据库及隔离环境已Git-ignore，未提交。
'''
    write_text(ROOT/'RAG_EVALUATION_FINAL_REPORT.md',report)
    print(json.dumps({'original_database_unchanged':original_unchanged,'budget':budget,'test_result':test_result},ensure_ascii=False))


if __name__=='__main__': main()
