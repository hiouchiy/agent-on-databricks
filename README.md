# 医療アシスタントAIエージェント ハンズオンラボ

Databricks上で動作する医療AIエージェントの実装ハンズオン。ReACTの基礎から本番デプロイまでを段階的に学習します。

## 🎯 概要

LangGraph、Unity Catalog、Vector Searchを活用した実用的なAIエージェント。

### 主な機能
- 医療マニュアル検索（RAG）
- 乳がんリスク予測
- Unity Catalog関数統合
- MLflow自動評価
- Model Servingデプロイ

## 🏗️ アーキテクチャ

```
ユーザー → AIエージェント (LangGraph)
              ├─→ Vector Search (医療マニュアル)
              ├─→ UC関数: get_patient_data
              └─→ UC関数: predict_cancer
```

## 📦 必要要件

- Unity Catalog、Vector Search、Model Serving有効化
- DBR 14.3 LTS以上
- LLMエンドポイント（databricks-llama-4-maverickなど）

## 🚀 セットアップ

### 1. データ準備

`data/` フォルダにPDFを配置：
- 胃がん治療マニュアル.pdf
- 食道がん治療マニュアル.pdf
- 乳がん治療マニュアル.pdf
- 前立腺がん治療マニュアル.pdf

### 2. ノートブック実行（順番に）

| # | ノートブック | 所要時間 | 目的 |
|---|------------|---------|------|
| 01 | **data_prep.py** | 10分 | PDFチャンク化、Vector Index作成、患者データ生成 |
| 02 | **tool_prep.py** | 5分 | Unity Catalog関数作成（3つ） |
| 03 | **simple_react_agent.py** | 15分 | ReACT動作原理を手動実行で理解 |
| 04 | **agent_develop.py** | 15分 | LangGraphエージェント構築と評価 |
| 05 | **agent_deploy.py** | 20分 | MLflow登録とModel Servingデプロイ |

## 📖 使い方

### ノートブック内
```python
from simple_agent import AGENT

response = AGENT.invoke({
    "messages": [{"role": "user", "content": "乳がんの治療法は？"}]
})
print(response["messages"][-1].content)
```

### REST API（デプロイ後）
```python
import requests
response = requests.post(
    f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": "質問"}]}
)
```

## 📁 プロジェクト構成

```
├── README.md
├── data/                          # 医療マニュアルPDF
├── 01.data_prep.py               # データ準備
├── 02.tool_prep.py               # ツール作成
├── 03.simple_react_agent.py      # ReACT理解
├── 04.agent_develop.py           # エージェント開発・評価
├── 05.agent_deploy.py            # 登録・デプロイ
└── simple_agent.py               # エージェント実装
```

## 🛠️ 技術スタック

- **AI**: LangChain, LangGraph
- **Platform**: Databricks (Unity Catalog, Vector Search, Model Serving)
- **ML/Data**: scikit-learn, MLflow, Delta Lake
- **Processing**: pdfplumber, pysbd, transformers

## 🔧 カスタマイズ

- **システムプロンプト**: `simple_agent.py`の`SYSTEM_PROMPT`を編集
- **ツール追加**: `02.tool_prep.py`で関数作成 → `simple_agent.py`に追加
- **評価データ追加**: `04.agent_develop.py`の`eval_data`に追加

## 🐛 トラブルシューティング

| 問題 | 解決方法 |
|------|----------|
| Vector Indexエラー | エンドポイント作成を確認 |
| UC関数が見つからない | カタログ・スキーマ作成を確認 |
| 応答が遅い | `num_results`を減らす |

## 📚 参考リソース

- [Databricks エージェント評価](https://docs.databricks.com/aws/ja/generative-ai/agent-evaluation)
- [MLflow Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

## ⚠️ 注意事項

教育目的のデモです。実際の医療現場での使用には適切な医療監修と法規制への準拠が必要です。

---

**License**: MIT | **Made with ❤️ on Databricks**