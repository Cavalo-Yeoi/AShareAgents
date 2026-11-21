# AShareAgents

> 🤖 面向中国 A 股市场的多智能体（Multi-Agent）投研与决策辅助平台。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

AShareAgents 基于现代 Agent 架构设计，融合 Multi-Agent 协同、RAG（检索增强生成）、长期记忆、上下文工程、金融知识库与工作流编排等核心能力，实现对市场、行业、公司、财报、新闻、公告、风险以及量化数据的自动化分析与研究，帮助用户构建真正具备自主研究能力的 AI 投研系统。
## 🚀 快速开始

### 项目安装
1. 克隆 AShareAgents：
```bash
git clone https://github.com/Cavalo-Yeoi/AShareAgents.git
```
2. 进入项目目录：
```bash
cd AShareAgents
```
3. 创建虚拟环境（以 uv 为例）：
```bash
uv venv
```
4. 激活环境：
```bash
# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```
5. 安装包及其依赖项：
```bash
uv pip install -e .
```

### 环境配置
创建 `.env` 文件：
```bash
cp .env.example .env
```
最少提供一个模型 API KEY。 建议配置多个模型 API KEY，前端可以根据需求进行选择。

### 项目启动
FastAPI 后端：
```bash
uvicorn AShareAgents.api.app:app --host 127.0.0.1 --port 8000
```

Streamlit 前端：
```bash
streamlit run frontend/app.py
```

### 支持的 LLM 提供商

| 提供商           | 接入方式 / 示例模型                  | 环境变量 |
|---------------|------------------------------| --- |
| OpenAI        | GPT 系列 / 原生 Responses API    | `OPENAI_API_KEY` |
| Anthropic     | Claude 系列                    | `ANTHROPIC_API_KEY` |
| Google Gemini | Gemini 系列（需安装 `google` 可选依赖） | `GOOGLE_API_KEY` |
| DeepSeek      | DeepSeek Chat / Reasoner     | `DEEPSEEK_API_KEY` |
| Qwen          | DashScope OpenAI 兼容接口        | `DASHSCOPE_API_KEY` |
| GLM           | 智谱 OpenAI 兼容接口               | `ZHIPU_API_KEY` |
| MiniMax       | MiniMax OpenAI 兼容接口          | `MINIMAX_API_KEY` |
| xAI           | Grok 系列                      | `XAI_API_KEY` |


## 🏗️ 项目结构
```
AShareAgents/
│
├── AShareAgents/                  # 核心 Python 业务包
│   │
│   ├── agents/                    # 多智能体分析模块
│   ├── api/                       # API 接口模块
│   ├── datasource/                # A 股行情及财经数据源
│   │   ├── astock/                # A股数据聚合、回退与格式化
│   │   ├── alphaVantage/          # Alpha Vantage
│   │   ├── yFinance/              # Yahoo Finance
│   │   ├── eastMoney/             # 东方财富
│   │   ├── mootdx/                # 通达信行情
│   │   ├── sinaFinance/           # 新浪财经
│   │   ├── tencentFinance/        # 腾讯财经
│   │   ├── tongHuaShun/           # 同花顺
│   │   ├── clsFinance/            # 财联社
│   │   └── baiduFinance/          # 百度股市通
│   │
│   ├── llm/                       # LLM 客户端与模型配置
│   ├── memory/                    # 智能体记忆模块
│   ├── models/                    # 结构化数据模型
│   ├── rag/                       # RAG 检索模块
│   │
│   ├── evaluation/                # RAG 评测模块
│   │   ├── configs/               # 评测配置
│   │   ├── scripts/               # 评测执行脚本
│   │   ├── ingestion.py           # 文档处理与入库
│   │   ├── retrieval.py           # 检索策略
│   │   ├── generation.py          # 生成评测相关逻辑
│   │   ├── metrics.py             # 评测指标
│   │   └── io.py                  # 评测文件与路径管理
│   │
│   ├── storage/                   # 存储模块
│   ├── tools/                     # Agent 工具
│   ├── workflows/                 # LangGraph 工作流
│   ├── config.py                  # 项目配置
│   └── logging_config.py          # 日志配置
│
├── main.py                        # 项目入口
├── pyproject.toml                 # 项目与依赖配置
├── requirements.txt               # Python 依赖清单
├── .gitignore                     # Git 忽略规则
└── README.md                      # 项目说明
```
