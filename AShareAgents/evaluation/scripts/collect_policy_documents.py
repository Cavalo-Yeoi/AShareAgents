"""Collect public official policy originals, preserving response bytes and evidence.

Run with existing environment. No API keys, paid services, or database access.
HTML is retained as HTML; derived body text is clearly marked and never counted.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/rag_eval_v1/raw/policy'
MAN = ROOT / 'data/rag_eval_v1/manifests'
PUBLICATION_LABEL = r'(?:发布时间|发布日期|发布日\s*期|(?<![\u4e00-\u9fff])日期|(?<![\u4e00-\u9fff])时间)'

# URLs discovered through primary-source web search, not inferred from URL patterns.
# Expected title terms are checked against actual response body before acceptance.
SEEDS = [
 ('capital_market', '国务院关于加强监管防范风险推动资本市场高质量发展的若干意见', '证券市场', '国务院', 'https://www.csrc.gov.cn/csrc/c100028/c7473562/content.shtml'),
 ('listed_supervision', '关于加强上市公司监管的意见（试行）', '上市公司监管', '中国证券监督管理委员会', 'https://www.csrc.gov.cn/csrc/c100028/c7467850/content.shtml'),
 ('stock_repurchase', '关于设立股票回购增持再贷款有关事宜的通知', '货币政策/股票回购增持', '中国人民银行;国家金融监督管理总局;中国证券监督管理委员会', 'https://www.csrc.gov.cn/csrc/c100028/c7513043/content.shtml'),
 ('share_reduction', '上市公司股东减持股份管理暂行办法', '上市公司股东减持', '中国证券监督管理委员会', 'https://www.csrc.gov.cn/csrc/c101953/c7483190/content.shtml'),
 ('equipment_upgrade', '推动大规模设备更新和消费品以旧换新行动方案', '宏观/设备更新/耐用消费品', '国务院', 'https://www.mee.gov.cn/zcwj/gwywj/202403/t20240318_1068594.shtml'),
 ('energy_saving', '2024—2025年节能降碳行动方案', '能源/工业节能降碳', '国务院', 'https://www.mee.gov.cn/zcwj/gwywj/202405/t20240530_1074495.shtml'),
 ('carbon_control', '加快构建碳排放双控制度体系工作方案', '碳排放/能源管理', '国务院办公厅', 'https://www.mee.gov.cn/zcwj/gwywj/202408/t20240806_1083433.shtml'),
 ('venture_capital', '促进创业投资高质量发展的若干政策措施', '创业投资/科技金融', '国务院办公厅', 'https://www.mee.gov.cn/zcwj/gwywj/202406/t20240620_1076246.shtml'),
 ('service_consumption', '国务院关于促进服务消费高质量发展的意见', '服务消费/文旅/养老', '国务院', 'https://www.mee.gov.cn/zcwj/gwywj/202408/t20240806_1083436.shtml'),
 ('consumption_boost', '提振消费专项行动方案', '宏观/扩大内需/消费', '中共中央办公厅;国务院办公厅', 'https://www.mee.gov.cn/zcwj/zyygwj/202503/t20250316_1104111.shtml'),
 ('pharma_reform', '全面深化药品医疗器械监管改革促进医药产业高质量发展的意见', '医药/医疗器械', '国务院办公厅', 'https://www.mee.gov.cn/zcwj/gwywj/202501/t20250106_1100064.shtml'),
 ('chinese_medicine', '提升中药质量促进中医药产业高质量发展的意见', '中药/中医药产业', '国务院办公厅', 'https://app.www.gov.cn/govdata/gov/202503/20/525664/article.html'),
 ('silver_economy', '发展银发经济增进老年人福祉的意见', '养老/银发经济', '国务院办公厅', 'https://app.www.gov.cn/govdata/gov/202401/15/511180/article.html'),
 ('urbanization', '深入实施以人为本的新型城镇化战略五年行动计划', '宏观/城镇化/基础设施', '国务院', 'https://app.www.gov.cn/govdata/gov/202407/31/517833/article.html'),
 ('equipment_fiscal', '关于实施设备更新贷款财政贴息政策的通知', '财政/设备更新贷款贴息', '财政部;国家发展改革委;中国人民银行;金融监管总局', 'https://www.mof.gov.cn/jrttts/202406/t20240626_3937996.htm'),
 ('tradein_fiscal', '关于加力支持大规模设备更新和消费品以旧换新的若干措施', '财政/设备更新/消费品以旧换新', '国家发展改革委;财政部', 'https://www.ndrc.gov.cn/xxgk/zcfb/tz/202407/t20240725_1391941_ext.html'),
 ('consumer_tradein', '推动消费品以旧换新行动方案', '汽车/家电/家装消费', '商务部等14部门', 'https://www.ndrc.gov.cn/xwdt/ztzl/tddgmsbgxhxfpyjhx/gzdt/202404/t20240417_1365730_ext.html'),
 ('ic_tax', '关于做好2024年享受税收优惠政策的集成电路企业或项目、软件企业清单制定工作有关要求的通知', '半导体/集成电路/软件税收', '国家发展改革委;工业和信息化部;财政部;海关总署;税务总局', 'https://www.ndrc.gov.cn/xwdt/tzgg/202403/t20240322_1365170.html'),
 ('ic_vat', '关于2024年度享受增值税加计抵减政策的集成电路企业清单制定工作有关要求的通知', '半导体/集成电路增值税', '工业和信息化部;国家发展改革委;财政部;税务总局', 'https://www.miit.gov.cn/jgsj/dzs/wjfb/art/2024/art_8e41a65ffc014c54b16e56821907c262.html'),
 ('nev_procurement', '关于进一步明确新能源汽车政府采购比例要求的通知', '新能源汽车/政府采购', '财政部办公厅', 'https://gks.mof.gov.cn/guizhangzhidu/202412/t20241230_3950760.htm'),
 ('nev_tax', '关于延续和优化新能源汽车车辆购置税减免政策的公告', '新能源汽车/车辆购置税', '财政部;税务总局;工业和信息化部', 'https://m.mof.gov.cn/czxw/202306/t20230620_3891500.htm'),
 ('lithium_battery', '锂离子电池行业规范条件（2024年本）', '锂离子电池/新能源产业', '工业和信息化部', 'https://www.miit.gov.cn/zwgk/zcwj/wjfb/gg/art/2024/art_dfe849c6837c4a50bf3e3c30d1697710.html'),
 ('bank_capital', '商业银行资本管理办法', '银行/资本监管', '国家金融监督管理总局', 'https://www.moj.gov.cn/pub/sfbgw/flfggz/flfggzbmgz/202409/t20240909_505612.html'),
 ('bank_advanced', '商业银行实施资本计量高级方法申请及验收规定', '银行/资本计量', '国家金融监督管理总局', 'https://app.www.gov.cn/govdata/gov/202411/03/521277/article.html'),
 ('fair_competition', '公平竞争审查条例实施办法', '市场监管/公平竞争', '国家市场监督管理总局', 'https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2025/art_2084c3ba225943c2a670d27e85fb00be.html'),
]

def normalized(text):
    return re.sub(r'[\s\u3000—－–（）()]', '', text)

def date_value(value):
    value = unicodedata.normalize('NFKC', value)
    m = re.search(r'(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})', value)
    if m:
        return f'{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}'
    return None

def collect(seed):
    key, title, industry, issuer, url = seed
    dest = RAW / key
    dest.mkdir(parents=True, exist_ok=True)
    events = []
    def fetch(address):
        try:
            r = requests.get(address, timeout=(15, 45), headers={'User-Agent': 'Mozilla/5.0 (compatible; public-policy-research/1.0)'})
            events.append({'url': address, 'final_url': r.url, 'status_code': r.status_code, 'bytes': len(r.content)})
            r.raise_for_status()
            return r
        except Exception as exc:
            events.append({'url': address, 'error': type(exc).__name__ + ': ' + str(exc)})
            raise
    try:
        try:
            response = fetch(url)
        except requests.RequestException:
            if key == 'ic_vat':
                url = 'https://tianjin.chinatax.gov.cn/11200000000/0300/030004/03000418/20241017102940934.shtml'
                # Fallbacks must be discovered official URLs, never guessed.
                response = fetch(url)
            else:
                raise
        (dest/'source.html').write_bytes(response.content)
        soup = BeautifulSoup(response.content, 'html.parser')
        metas = {str(m.get('name') or m.get('property') or '').lower(): m.get('content', '') for m in soup.find_all('meta')}
        full_text = soup.get_text('\n', strip=True)
        selector = None
        body = None
        for sel in ['.TRS_Editor', '.trs_editor', '.Custom_UnionStyle', '#UCAP-CONTENT', '.content_detail', '.content', '.article', '#zoom', 'article']:
            candidate = soup.select_one(sel)
            if candidate and len(candidate.get_text(strip=True)) > 200:
                selector, body = sel, candidate.get_text('\n',strip=True)
                break
        body = body or full_text
        if normalized(title) not in normalized(full_text):
            raise ValueError('Expected policy title not present in actual page')
        if len(body) < 250 or len(re.findall('[\u4e00-\u9fff]', body)) < 120:
            raise ValueError('No substantive Chinese policy text')
        publish_date, evidence = None, None
        # Visible publication labels outrank generic meta tags: CSRC/SAMR pages
        # observed in this collection have migration timestamps in PubDate.
        visible_date = re.search(PUBLICATION_LABEL + r'[：:\s]*(20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}(?:日)?)', full_text[:2500])
        if visible_date:
            publish_date, evidence = date_value(visible_date[1]), visible_date[0]
        for field in ['pubdate','publishdate','pubdate','dc.date','date','article:published_time','pubdate','firstpublishedtime']:
            if not publish_date and date_value(metas.get(field, '')):
                publish_date = date_value(metas[field])
                evidence = f'meta[{field}] content={metas[field]}'
                break
        if not publish_date:
            m = re.search(PUBLICATION_LABEL + r'[：:\s]*(20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}(?:日)?)', full_text[:2500])
            if m:
                publish_date, evidence = date_value(m[1]), m[0]
        if not publish_date:
            # Inspect first page header date only; document signature dates are kept separate.
            head = full_text[:max(1000, full_text.find(title)+len(title)+300)]
            m = re.search(r'20\d{2}-\d{2}-\d{2}', head)
            if m:
                publish_date, evidence = m[0], 'page_header: ' + head[max(0,m.start()-60):m.end()+60]
        if key in {'share_reduction','nev_tax'}:
            # Generic CSRC PubDate can be a migration date; mobile MOF lacks
            # metadata. Preserve supplemental official publication evidence.
            evidence_url = ('https://www.csrc.gov.cn/csrc/c100028/c7483136/content.shtml' if key=='share_reduction' else 'https://www.miit.gov.cn/jgsj/zbys/wjfb/art/2023/art_eda11413329c4071ac3257da3615ed3b.html')
            try:
                er=fetch(evidence_url)
                (dest/'publication_evidence.html').write_bytes(er.content)
                es=BeautifulSoup(er.content,'html.parser').get_text(' ',strip=True)
                ed=re.search(PUBLICATION_LABEL+r'[：:\s]*(20\d{2}-\d{1,2}-\d{1,2})',es[:2500])
                if ed and normalized(title) in normalized(es):
                    publish_date, evidence = date_value(ed[1]), f'{evidence_url} visible publication metadata: {ed[0]}'
                elif key == 'share_reduction':
                    publish_date, evidence = None, None
            except requests.RequestException:
                if key == 'share_reduction':
                    publish_date, evidence = None, None
        if key == 'share_reduction':
            # The rule's proclamation explicitly dates its public announcement;
            # generic HTML PubDate is a later site migration timestamp.
            compact = re.sub(r'\s+', '', full_text)
            proclamation = re.search(r'现予公布，?自公布之日起施行.*?主席.*?(20\d{2}年\d{1,2}月\d{1,2}日)', compact)
            if proclamation:
                publish_date, evidence = date_value(proclamation[1]), 'official proclamation: '+proclamation[0]
        links = [{'text': a.get_text(' ',strip=True), 'url': urljoin(response.url,a.get('href',''))} for a in soup.find_all('a',href=True) if '.pdf' in a['href'].lower()]
        pdf_links = [a for a in links if normalized(title) in normalized(a['text']) or ('减持' in title and '减持' in a['text'] and '说明' not in a['text']) or a['text']=='下载pdf版']
        if key == 'lithium_battery':
            pdf_links = [a for a in links if '规范条件' in a['text']]
        raw_path, raw_bytes, file_format = dest/'source.html', response.content, 'html'
        attachment_text = ''
        for item in pdf_links[:1]:
            try:
                pr = fetch(item['url'])
                if not pr.content.startswith(b'%PDF'):
                    raise ValueError('Attachment is not PDF magic')
                reader = PdfReader(io.BytesIO(pr.content))
                attachment_text = '\n\n'.join(p.extract_text() or '' for p in reader.pages)
                if len(attachment_text) < 200:
                    raise ValueError('PDF has no substantive extractable policy text')
                raw_path, raw_bytes, file_format = dest/'original.pdf', pr.content, 'pdf'
                raw_path.write_bytes(raw_bytes)
                break
            except Exception as exc:
                events.append({'attachment_validation_error':str(exc), 'url':item['url']})
        effective = None
        effective_text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', attachment_text + '\n' + body))
        em = re.search(r'自(20\d{2}年\d{1,2}月\d{1,2}日)起(?:施行|实施)', effective_text)
        if em:
            effective = date_value(em[1])
        elif '自公布之日起施行' in body or '自发布之日起施行' in body:
            effective = publish_date
        (dest/'body.txt').write_text(attachment_text or body,encoding='utf-8')
        issuer_terms = [x for x in issuer.split(';') if x in full_text or x in attachment_text]
        if not issuer_terms:
            raise ValueError('Issuer not evidenced by actual policy page/body')
        signature_dates = re.findall(r'20\d{2}年\s*\d{1,2}月\s*\d{1,2}日', attachment_text or body)
        info = {'source_url':url,'final_url':response.url,'metadata':metas,'publication_metadata_conflict':bool(publish_date and date_value(metas.get('pubdate','')) and publish_date!=date_value(metas['pubdate'])),'body_selector':selector,'body_extract_is_derivative':True,'html_source_preserved':'source.html','pdf_links_found':links,'selected_pdf_links':pdf_links,'original_pdf_status': 'downloaded_original' if file_format=='pdf' else 'no_policy_original_pdf_link_on_official_page' if not pdf_links else 'official_pdf_download_failed_html_original_preserved','issuer_evidence_terms':issuer_terms,'signature_or_intext_dates':signature_dates,'effective_date_evidence':em[0] if em else None,'public_text_explicitly_abridged':'本文有删减' in full_text,'body_characters':len(attachment_text or body)}
        (dest/'evidence.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        record = dict(document_id='policy_'+key,title=title,knowledge_type='policy_industry',document_type='policy',ticker='',company_name='',industry=industry,issuer=issuer,source_url=url,source_level='official',publish_date=publish_date,report_period='',effective_date=effective,expire_date=None,revision_date=None,downloaded_at=datetime.now(timezone.utc).isoformat(),original_filename=unquote(urlparse(pdf_links[0]['url'] if file_format=='pdf' else response.url).path.rsplit('/',1)[-1]),file_sha256=hashlib.sha256(raw_bytes).hexdigest(),raw_path=raw_path.relative_to(ROOT).as_posix(),date_status='verified' if publish_date else 'unknown',date_verification_method='official_publication_metadata' if publish_date else None,date_evidence_url=url,date_evidence_text=evidence,collection_status='collected',file_format=file_format,body_text_path=(dest/'body.txt').relative_to(ROOT).as_posix(),evidence_path=(dest/'evidence.json').relative_to(ROOT).as_posix(),pdf_original_status=info['original_pdf_status'])
        if evidence and evidence.startswith('https://'):
            record['date_evidence_url'] = evidence.split(' ')[0]
        record['download_url'] = pdf_links[0]['url'] if file_format=='pdf' else response.url
        return record, events
    except Exception as exc:
        return {'document_id':'policy_'+key,'title':title,'source_url':url,'collection_status':'failed','error':type(exc).__name__+': '+str(exc)}, events

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    RAW.mkdir(parents=True,exist_ok=True); MAN.mkdir(parents=True,exist_ok=True)
    records, failed, logs = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record, events in pool.map(collect,SEEDS):
            (records if record['collection_status']=='collected' else failed).append(record)
            logs.extend({'document_id':record['document_id'],**e} for e in events)
            print(json.dumps({k:record.get(k) for k in ['document_id','publish_date','file_format','collection_status','error']},ensure_ascii=True),flush=True)
    seen = set()
    for r in records:
        if r['file_sha256'] in seen:
            raise ValueError('Duplicate raw policy hash')
        seen.add(r['file_sha256'])
    for name, rows in [('policy_documents.jsonl',records),('policy_failures.jsonl',failed),('policy_download_log.jsonl',logs)]:
        if name != 'policy_documents.jsonl' and (MAN/name).exists():
            rows = [json.loads(line) for line in (MAN/name).read_text(encoding='utf-8').splitlines() if line] + rows
        (MAN/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    if records:
        with (MAN/'policy_documents.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    print(json.dumps({'collected':len(records),'failed':len(failed),'unknown_dates':sum(r['publish_date'] is None for r in records)}))

if __name__ == '__main__':
    main()
