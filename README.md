# 医療アシスタントAIエージェント

Databricks上で動作する、医療マニュアル検索と乳がんリスク予測機能を持つAIエージェントのハンズオン実装です。

## 🎯 概要

LangChain、Unity Catalog、Databricks Vector Searchを活用した実用的な医療AIエージェントです。RAG（検索拡張生成）と機械学習モデルを統合し、医療従事者や患者からの質問に対して的確な回答を提供します。

### 主な機能

- **医療マニュアル検索（RAG）**: ベクトル検索で最新のがん治療情報を提供
- **乳がんリスク予測**: 機械学習モデルによる悪性/良性の判定
- **Unity Catalog統合**: 患者データ取得と予測をツールとして統合
- **自動評価**: MLflowによる品質評価とデプロイ

## 🏗️ アーキテクチャ

```
ユーザー
  ↓
医療アシスタントAIエージェント (LangGraph)
  ├─→ Vector Search (医療マニュアル検索)
  ├─→ UC関数 (患者データ取得)
  └─→ Model Serving (乳がん予測)
```

## 📦 必要要件

### Databricks環境
- Unity Catalog有効化済み
- Vector Search & Model Serving利用可能
- DBR 14.3 LTS以上推奨

### LLMエンドポイント
- `databricks-gpt-oss-20b` または `databricks-llama-4-maverick`

### 主要ライブラリ
```
databricks-langchain
unitycatalog-ai[databricks]
langchain / langgraph
mlflow
```

## 🚀 セットアップ

### 1. データの準備

`data/` フォルダには以下のがん治療マニュアルPDFが含まれています：

```
data/
├── 胃がん治療マニュアル.pdf
├── 食道がん治療マニュアル.pdf
├── 乳がん治療マニュアル.pdf
└── 前立腺がん治療マニュアル.pdf
```

これらのPDFは自動的にチャンク分割され、ベクトルインデックス化されます。

### 2. ノートブックの実行（順番に実行）

#### **01.data_prep.py** (約10分)
PDFからテキスト抽出、チャンク分割、Vector Searchインデックス作成、患者データ生成

#### **02.tool_prep.py** (約5分)
Unity Catalog関数の作成（`get_patient_data`, `predict_cancer`）

#### **03.agent_develop.py** (約5分)
エージェントの構築とテスト

#### **04.agent_eval.py** (約15分)
評価実行、MLflow登録、Model Servingデプロイ

## 📖 使い方

### エージェントとの対話

```python
from simple_agent import AGENT

# 医療マニュアル検索
response = AGENT.invoke({
    "messages": [{
        "role": "user",
        "content": "乳がんの治療法について教えてください"
    }]
})

# 乳がんリスク予測
response = AGENT.invoke({
    "messages": [{
        "role": "user",
        "content": "P000123の患者は悪性の可能性はありますか？"
    }]
})
```

### REST API経由での利用

```python
import requests

response = requests.post(
    "https://your-workspace.cloud.databricks.com/serving-endpoints/medical_agent/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": "質問内容"}]}
)
```

## 📁 プロジェクト構成

```
medical-assistant-agent/
├── README.md
├── data/                    # 医療マニュアルPDFフォルダ
│   └── (4種類のがん治療マニュアル)
├── 01.data_prep.py         # データ準備
├── 02.tool_prep.py         # ツール作成
├── 03.agent_develop.py     # エージェント開発
├── 04.agent_eval.py        # 評価とデプロイ
└── simple_agent.py         # エージェント実装
```

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|---------|------|
| AIフレームワーク | LangChain, LangGraph |
| プラットフォーム | Databricks (Unity Catalog, Vector Search, Model Serving) |
| ML/データ | scikit-learn, Delta Lake, MLflow |
| 処理 | pdfplumber, pysbd, transformers |

## 📊 データについて

### 医療マニュアル
- **内容**: 胃がん、食道がん、乳がん、前立腺がんの治療法
- **形式**: PDF（日本語）
- **処理**: 800トークンチャンク、160トークンオーバーラップ

### 乳がんデータセット
- **出典**: scikit-learn Wisconsin Breast Cancer Dataset
- **サンプル数**: 1,000件
- **特徴量**: 30次元（細胞核の特徴）

## 🔧 カスタマイズ

### システムプロンプトの変更
`simple_agent.py`の`SYSTEM_PROMPT`を編集

### ツールの追加
`02.tool_prep.py`で新しいUnity Catalog関数を作成し、`UC_TOOL_NAMES`に追加

### 評価データの拡張
`04.agent_eval.py`の`eval_data`リストに新しいテストケースを追加

## 🐛 トラブルシューティング

| 問題 | 解決方法 |
|------|---------|
| Vector Searchインデックスエラー | エンドポイントの作成と権限を確認 |
| Unity Catalog関数が見つからない | カタログ・スキーマの作成を確認 |
| エージェントの応答が遅い | `num_results`を減らす、エンドポイントをスケール |

## 📝 ライセンス

MIT License

## ⚠️ 注意事項

このプロジェクトは教育目的のデモンストレーションです。実際の医療現場での使用には、適切な医療監修と法規制への準拠が必要です。

---

**質問・提案**: GitHubのIssueをご利用ください