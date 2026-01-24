# Databricks notebook source
# MAGIC %md
# MAGIC # AIエージェントの評価・登録・デプロイハンズオンラボ
# MAGIC
# MAGIC ## このノートブックで学ぶこと
# MAGIC
# MAGIC このハンズオンでは、前のノートブック（04.agent_develop）で構築した
# MAGIC 医療アシスタントエージェントを、本番環境で使えるようにする方法を学びます。
# MAGIC
# MAGIC 具体的には、以下の3つのフェーズを実施します：
# MAGIC
# MAGIC ### フェーズ1: エージェントの評価
# MAGIC
# MAGIC エージェントが期待通りに動作しているかを客観的に測定します。
# MAGIC
# MAGIC - エージェントは正確な情報を提供しているか？
# MAGIC - 検索した情報を適切に活用しているか？
# MAGIC - 回答は質問に対して適切か？
# MAGIC - 安全性やバイアスの問題はないか？
# MAGIC
# MAGIC ### フェーズ2: モデルの登録と管理
# MAGIC
# MAGIC エージェントをMLflowモデルとして登録し、Unity Catalogで管理します。
# MAGIC
# MAGIC - バージョン管理による変更履歴の追跡
# MAGIC - リネージによる依存関係の可視化
# MAGIC - チーム間での共有とアクセス制御
# MAGIC - 再現可能な環境の確保
# MAGIC
# MAGIC ### フェーズ3: 本番環境へのデプロイ
# MAGIC
# MAGIC エージェントをModel ServingエンドポイントとしてデプロイしてREST APIで公開します。
# MAGIC
# MAGIC - スケーラブルなAPI環境の構築
# MAGIC - レビューアプリでのチーム共有
# MAGIC - 本番環境でのテストと監視
# MAGIC - 外部アプリケーションからの呼び出し
# MAGIC
# MAGIC ### このノートブックの流れ
# MAGIC
# MAGIC #### 準備フェーズ（ステップ1-6）
# MAGIC 1. **環境準備**: 必要なライブラリのインストール
# MAGIC 2. **環境設定**: カタログとスキーマの設定
# MAGIC 3. **環境変数設定**: simple_agent.py用の設定
# MAGIC 4. **エージェント読み込み**: simple_agent.pyからエージェントを読み込む
# MAGIC 5. **動作確認**: エージェントが正しく動作するかテスト
# MAGIC 6. **ワークフロー可視化**: エージェントの内部構造を図で確認
# MAGIC
# MAGIC #### フェーズ1: エージェントの評価（ステップ7-10）
# MAGIC 7. **MLflowモデル登録**: エージェントをMLflowモデルとして登録
# MAGIC 8. **評価用ラッパー関数定義**: MLflow評価用の関数を作成
# MAGIC 9. **評価データセット作成**: テスト用の質問と期待される回答を準備
# MAGIC 10. **スコアラー定義**: 評価指標（Correctness、RetrievalSufficiencyなど）を設定
# MAGIC 11. **再評価の実行**: 登録したモデルで評価を実行（オプション）
# MAGIC
# MAGIC #### フェーズ2: モデルの登録と管理（ステップ12-13）
# MAGIC 12. **Unity Catalog登録**: モデルをUnity Catalogに登録
# MAGIC 13. **リネージ確認**: モデルの依存関係を可視化
# MAGIC
# MAGIC #### フェーズ3: 本番環境へのデプロイ（ステップ14-15）
# MAGIC 14. **エージェントデプロイ**: Model Servingエンドポイントとして公開
# MAGIC 15. **デプロイテスト**: デプロイしたエージェントをREST APIで呼び出し
# MAGIC
# MAGIC #### まとめ（ステップ16）
# MAGIC 16. **振り返りと次のステップ**: 学んだことの整理と今後の展開
# MAGIC
# MAGIC ### simple_agent.pyファイルについて
# MAGIC
# MAGIC このノートブックは、別ファイル（`simple_agent.py`）に定義されたエージェントを
# MAGIC 読み込んで評価・登録・デプロイします。
# MAGIC
# MAGIC #### simple_agent.pyに含まれる内容
# MAGIC
# MAGIC - LLMの設定（環境変数から読み込み）
# MAGIC - ベクトル検索ツールの設定
# MAGIC - Unity Catalog関数ツールの設定
# MAGIC - システムプロンプトの定義
# MAGIC - エージェントの構築ロジック
# MAGIC
# MAGIC #### この構成のメリット
# MAGIC
# MAGIC - **コードの分離**: エージェントのロジックと評価・デプロイコードを分離
# MAGIC - **再利用性**: 同じエージェントコードを異なる環境で使用可能
# MAGIC - **バージョン管理**: エージェントの変更履歴を明確に管理
# MAGIC - **デプロイ容易性**: ファイル単位でモデルとして登録・デプロイ可能
# MAGIC
# MAGIC #### 環境変数による設定
# MAGIC
# MAGIC simple_agent.pyは以下の環境変数を参照します：
# MAGIC - `LLM_ENDPOINT_NAME`: 使用するLLMエンドポイント
# MAGIC - `VS_NAME`: ベクトル検索インデックス名
# MAGIC - `UC_TOOL_NAMES`: Unity Catalog関数のパターン
# MAGIC
# MAGIC これにより、コードを変更せずに異なる環境（開発・本番など）で使用できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境準備
# MAGIC
# MAGIC ### 必要なライブラリのインストール
# MAGIC
# MAGIC 以下のライブラリをインストールします：
# MAGIC
# MAGIC - **langgraph**: エージェントのワークフロー構築用（バージョン固定）
# MAGIC - **databricks-langchain**: DatabricksとLangChainの連携用
# MAGIC - **databricks-agents**: Databricksのエージェント機能用
# MAGIC - **uv**: 高速なパッケージインストーラー

# COMMAND ----------

# Unity Catalog AI統合パッケージのインストール
%pip install unitycatalog-ai[databricks] unitycatalog-langchain[databricks]

# Databricks LangChain統合パッケージのインストール
%pip install mlflow databricks-langchain databricks-agents uv

# Pythonを再起動して新しいライブラリを読み込む
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: 環境設定
# MAGIC
# MAGIC エージェントが使用するリソース（カタログ、スキーマ、インデックス）を
# MAGIC 環境変数として設定します。
# MAGIC
# MAGIC ### 環境変数を使う理由
# MAGIC
# MAGIC - **柔軟性**: コードを変更せずに設定を変更できる
# MAGIC - **セキュリティ**: 機密情報をコードに埋め込まない
# MAGIC - **再利用性**: 同じコードを異なる環境で使用できる
# MAGIC
# MAGIC ### simple_agent.pyとの連携
# MAGIC
# MAGIC ここで設定した環境変数は、agent.pyファイル内で参照されます。

# COMMAND ----------

# 現在のユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]

# ユーザー名を安全な形式に変換（@以降を削除し、記号をアンダースコアに置換）
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

# カタログとスキーマの名前を定義
CATALOG = f"exercise_{clean_username}"
SCHEMA = "agent"

# カタログとスキーマが存在しない場合は作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"✅ 環境セットアップ完了")
print(f"   カタログ: {CATALOG}")
print(f"   スキーマ: {SCHEMA}")

# カタログとスキーマの名前を変数に保存
catalog_name = CATALOG
system_schema_name = user_schema_name = SCHEMA

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: simple_agent.py用の環境変数設定
# MAGIC
# MAGIC simple_agent.pyファイルで使用する環境変数を設定します。
# MAGIC
# MAGIC ### 設定する環境変数
# MAGIC
# MAGIC - **LLM_ENDPOINT_NAME**: 使用するLLMのエンドポイント名
# MAGIC - **VS_NAME**: ベクトル検索インデックスの完全修飾名
# MAGIC - **UC_TOOL_NAMES**: Unity Catalog関数のパターン（ワイルドカード使用可）

# COMMAND ----------

import os

# ========================================
# 環境変数の設定
# ========================================

# LLMエンドポイント名
# ※環境に合わせて変更してください
os.environ["LLM_ENDPOINT_NAME"] = "databricks-gpt-oss-20b"  # または "databricks-llama-4-maverick"

# Vector Search インデックス名
# 01.Data-Prepで作成したベクトル検索インデックスを指定
os.environ["VS_NAME"] = f"{catalog_name}.{system_schema_name}.manuals_index"

# Unity Catalog関数ツール
# スキーマ内のすべての関数を使用（ワイルドカードパターン）
os.environ["UC_TOOL_NAMES"] = f"{catalog_name}.{user_schema_name}.*"

# 設定内容を確認
print("✅ 環境変数を設定しました:")
print(f"   LLM_ENDPOINT_NAME: {os.environ.get('LLM_ENDPOINT_NAME')}")
print(f"   VS_NAME: {os.environ.get('VS_NAME')}")
print(f"   UC_TOOL_NAMES: {os.environ.get('UC_TOOL_NAMES')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: simple_agent.pyからエージェントを読み込む
# MAGIC
# MAGIC ### simple_agent.pyファイルの役割
# MAGIC
# MAGIC agent.pyファイルには、エージェントの定義が含まれています。
# MAGIC このファイルは以下の処理を実行します：
# MAGIC
# MAGIC 1. 環境変数から設定を読み込む
# MAGIC 2. LLMを初期化
# MAGIC 3. ベクトル検索ツールを作成
# MAGIC 4. Unity Catalog関数ツールを読み込む
# MAGIC 5. システムプロンプトを定義
# MAGIC 6. エージェントを構築
# MAGIC
# MAGIC ### AGENTオブジェクト
# MAGIC
# MAGIC `AGENT`は、simple_agent.pyから読み込まれるエージェントのインスタンスです。
# MAGIC このオブジェクトを使って、質問に対する回答を生成できます。

# COMMAND ----------

# simple_agent.pyファイルからAGENTオブジェクトをインポート
from simple_agent import AGENT

print("✅ エージェントを読み込みました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: エージェントの動作確認
# MAGIC
# MAGIC 評価を実行する前に、エージェントが正しく動作するか確認します。

# COMMAND ----------

# テスト質問を実行
test_query = "P000123の患者は乳がんの悪性の可能性はありますか？"

# エージェントに質問を送信
res = AGENT.invoke({
    "messages": [{
        "role": "user", 
        "content": test_query
    }]
})

# 回答を表示
print(f"質問: {test_query}\n")
print(f"回答:\n{res["messages"][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: エージェントワークフローの可視化
# MAGIC
# MAGIC エージェントの内部構造を図で確認します。

# COMMAND ----------

from IPython.display import Image, display

try:
    # エージェントのグラフ構造を可視化
    display(Image(AGENT.get_graph().draw_mermaid_png()))
    print("✓ ワークフローの図を表示しました")
except Exception as e:
    print(f"図の表示に失敗しました: {e}")
    print("（この機能は環境によっては動作しない場合があります）")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # フェーズ1: エージェントの評価
# MAGIC ---
# MAGIC
# MAGIC ## 評価の重要性
# MAGIC
# MAGIC 本番環境にデプロイする前に、エージェントの品質を客観的に測定することが重要です。
# MAGIC
# MAGIC ### なぜ評価が必要なのか
# MAGIC
# MAGIC - **品質保証**: 様々なシナリオで一貫した品質を確保
# MAGIC - **問題の早期発見**: デプロイ前に不具合を検出
# MAGIC - **継続的改善**: システムプロンプトやツールの変更による影響を測定
# MAGIC - **ステークホルダーへの説明**: 数値で信頼性を証明
# MAGIC
# MAGIC まずは、エージェントをMLflowモデルとして登録し、評価の準備を整えます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: エージェントをMLflowモデルとして登録
# MAGIC
# MAGIC ### なぜMLflowに登録するのか
# MAGIC
# MAGIC エージェントをMLflowモデルとして登録すると、以下のメリットがあります：
# MAGIC
# MAGIC - **バージョン管理**: エージェントの異なるバージョンを管理
# MAGIC - **再現性**: 同じエージェントを後で再現できる
# MAGIC - **デプロイ**: Model Servingエンドポイントとしてデプロイ可能
# MAGIC - **依存関係の管理**: 必要なライブラリやリソースを一緒に保存
# MAGIC
# MAGIC ### PyFunc形式とは
# MAGIC
# MAGIC PyFunc（Python Function）は、MLflowの標準的なモデル形式です。
# MAGIC 任意のPythonコードをMLflowモデルとしてパッケージ化できます。
# MAGIC
# MAGIC ### simple_agent.pyファイルの登録
# MAGIC
# MAGIC `python_model="simple_agent.py"` を指定することで、
# MAGIC agent.pyファイルの内容がモデルとして登録されます。

# COMMAND ----------

from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
import simple_agent

# ========================================
# モデル登録の準備
# ========================================

# 入力例の定義（モデルの入力形式を示すサンプル）
input_example = {
    "messages": [{
        "role": "user",
        "content": "乳がんの治療方法について教えてください"
    }]
}

# リソースの定義（モデルが依存するファイルやディレクトリ）
# agent.pyが参照する可能性のあるリソースをリストアップ
resources = [DatabricksServingEndpoint(endpoint_name=simple_agent.LLM_ENDPOINT_NAME),
             DatabricksVectorSearchIndex(index_name=simple_agent.VS_NAME),
             DatabricksFunction(function_name=f"{catalog_name}.{user_schema_name}.get_patient_data"),
             DatabricksFunction(function_name=f"{catalog_name}.{user_schema_name}.predict_cancer"),
             DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en")]

# ========================================
# MLflowモデルとしてログに記録
# ========================================

print("📦 エージェントをMLflowモデルとして登録中...")

with mlflow.start_run(run_name="medical_agent_model_v1") as run:
    logged_agent_info = mlflow.langchain.log_model(
        name="agent",           # モデルの保存先（Run内のパス）
        lc_model="simple_agent.py",          # エージェントのコードファイル
        input_example=input_example,      # 入力例
        resources=resources,            # 依存リソース（必要に応じて有効化）
        extra_pip_requirements=[          # 追加のPython依存パッケージ
            "databricks-connect"
        ]
    )
    
    model_run_id = run.info.run_id

print(f"\n✅ モデルを登録しました")
print(f"   Run ID: {model_run_id}")
print(f"   Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: 評価用のラッパー関数を定義
# MAGIC
# MAGIC ### ラッパー関数の役割
# MAGIC
# MAGIC MLflowの評価フレームワークが期待する形式に、エージェントの入出力を
# MAGIC 変換する関数です。
# MAGIC
# MAGIC ### @mlflow.traceデコレーター
# MAGIC
# MAGIC このデコレーターを使うと、関数の実行がMLflowに自動的に記録されます。
# MAGIC 以下の情報がトレースされます：
# MAGIC
# MAGIC - 入力（質問）
# MAGIC - 出力（回答）
# MAGIC - 実行時間
# MAGIC - 使用したツール
# MAGIC - エラー情報（もしあれば）

# COMMAND ----------

import mlflow

@mlflow.trace
def predict_wrapper(query: str) -> str:
    """
    エージェントに質問を送信し、回答を返す関数
    
    Args:
        query: ユーザーからの質問（文字列）
    
    Returns:
        エージェントの回答（文字列）
    """
    # エージェントに質問を送信
    res = AGENT.invoke({
        "messages": [{
            "role": "user", 
            "content": query
        }]
    })
    
    # 最後のメッセージ（エージェントの回答）を返す
    return res["messages"][-1].content

print("✅ 評価用ラッパー関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9: 評価データセットの作成
# MAGIC
# MAGIC ### 評価データセットとは
# MAGIC
# MAGIC エージェントをテストするための質問と期待される回答のセットです。
# MAGIC 各テストケースには以下が含まれます：
# MAGIC
# MAGIC - **inputs**: テスト質問
# MAGIC - **expectations**: 期待される回答や事実
# MAGIC
# MAGIC ### 2種類の期待値
# MAGIC
# MAGIC 1. **expected_response**: 完全な回答文を指定
# MAGIC    - 回答が期待通りの内容かを評価
# MAGIC    
# MAGIC 2. **expected_facts**: 含まれるべき事実のリスト
# MAGIC    - 回答に特定の情報が含まれているかを評価
# MAGIC    - 複数の事実を個別にチェックできる
# MAGIC
# MAGIC ### データセットの設計ポイント
# MAGIC
# MAGIC - **多様性**: さまざまな種類の質問を含める
# MAGIC - **難易度**: 簡単な質問から複雑な質問まで
# MAGIC - **実用性**: 実際のユースケースに基づく質問
# MAGIC
# MAGIC ### 合成データの生成
# MAGIC
# MAGIC 評価データセットは手動で作成することもできますが、
# MAGIC [合成的に生成](https://www.databricks.com/jp/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities)
# MAGIC することも可能です。

# COMMAND ----------

import pandas as pd

# ========================================
# 評価データセットの定義
# ========================================

eval_data = [
    # ========================================
    # テストケース1: 基本的な治療方法の質問
    # ========================================
    {
        "inputs": {
            "query": "乳がんの治療には主にどんな方法がありますか？"
        },
        "expectations": {
            # 期待される完全な回答文
            "expected_response": "乳がんの治療には主に手術（外科治療）・放射線治療・薬物療法があり、診断されたときから緩和ケア／支持療法を受けることもできます。"
        }
    },
    
    # ========================================
    # テストケース2: 複雑な医療知識の質問
    # ========================================
    {
        "inputs": {
            "query": "食道がんの主な治療法は何ですか？"
        },
        "expectations": {
            # 回答に含まれるべき事実のリスト
            "expected_facts": [
                "主な治療法は、内視鏡的切除、手術、放射線治療、薬物療法、化学放射線療法である。",
                "診断されたときから、緩和ケア／支持療法を受けることができる。"
            ]
        }
    }
]

# DataFrameに変換して確認
eval_dataset = pd.DataFrame(eval_data)
print("✅ 評価データセットを作成しました")
print(f"   テストケース数: {len(eval_data)}")
display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ10: 評価指標（スコアラー）の定義
# MAGIC
# MAGIC ### スコアラーとは
# MAGIC
# MAGIC エージェントの回答を自動的に評価する「LLMジャッジ」です。
# MAGIC 別のLLMを使って、回答の品質を客観的にスコア化します。
# MAGIC
# MAGIC ### 利用可能なスコアラー
# MAGIC
# MAGIC MLflowには多数のスコアラーが用意されています：
# MAGIC
# MAGIC 1. **Correctness（正確性）**
# MAGIC    - 回答が期待される内容と一致しているかを評価
# MAGIC    - 事実の正確性をチェック
# MAGIC
# MAGIC 2. **RelevanceToQuery（関連性）**
# MAGIC    - 回答が質問に対して適切かを評価
# MAGIC    - 質問から逸脱していないかをチェック
# MAGIC
# MAGIC 3. **RetrievalGroundedness（根拠性）**
# MAGIC    - 回答が検索結果に基づいているかを評価
# MAGIC    - 幻覚（Hallucination）の検出
# MAGIC
# MAGIC 4. **RetrievalRelevance（検索関連性）**
# MAGIC    - 検索された文書が質問に関連しているかを評価
# MAGIC
# MAGIC 5. **RetrievalSufficiency（検索十分性）**
# MAGIC    - 検索された文書が回答に十分な情報を含むかを評価
# MAGIC
# MAGIC 6. **Safety（安全性）**
# MAGIC    - 回答に有害な内容が含まれていないかを評価
# MAGIC    - バイアスや差別的な表現をチェック
# MAGIC
# MAGIC ### スコアラーの選択
# MAGIC
# MAGIC 今回は、以下の2つのスコアラーを使用します：
# MAGIC - Correctness: 医療情報の正確性が重要
# MAGIC - RetrievalSufficiency: 検索が適切に機能しているかを確認
# MAGIC
# MAGIC 他のスコアラーはコメントアウトしていますが、
# MAGIC 必要に応じて有効化できます。

# COMMAND ----------

from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
)
import mlflow.genai

# ========================================
# 評価用スコアラーを定義
# ========================================

scorers = [
    Correctness(),              # 正確性を評価
    # RelevanceToQuery(),       # 質問への関連性を評価（コメントアウト）
    # RetrievalGroundedness(),  # 検索結果への根拠性を評価（コメントアウト）
    # RetrievalRelevance(),     # 検索結果の関連性を評価（コメントアウト）
    RetrievalSufficiency(),     # 検索結果の十分性を評価
    # Safety(),                 # 安全性を評価（コメントアウト）
]

print("✅ 評価スコアラーを定義しました")
print(f"   使用するスコアラー数: {len(scorers)}")
for scorer in scorers:
    print(f"   - {scorer.__class__.__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ11: 登録したモデルで評価（オプション）
# MAGIC
# MAGIC ### なぜ評価するのか
# MAGIC
# MAGIC 登録したMLflowモデルが正しく動作するか確認するため、
# MAGIC 再度評価を実行します。
# MAGIC
# MAGIC ### 最初の評価との違い
# MAGIC
# MAGIC - **最初の評価**: ノートブック上のコードを直接実行して評価
# MAGIC - **再評価**: MLflowに登録されたモデルを読み込んで評価
# MAGIC
# MAGIC これにより、モデルの登録とロードが正しく機能しているか確認できます。

# COMMAND ----------

# ========================================
# 登録したモデルをロード
# ========================================

print("📥 登録したモデルをロード中...")

# モデルURIを構築
logged_model_uri = f"runs:/{logged_agent_info.run_id}/agent"

# モデルをロード
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

print(f"✅ モデルをロードしました: {logged_model_uri}")

# ========================================
# ロードしたモデル用のラッパー関数を定義
# ========================================

def predict_wrapper_from_model(query: str) -> str:
    """
    ロードしたMLflowモデルを使って予測する関数
    
    Args:
        query: ユーザーからの質問（文字列）
    
    Returns:
        エージェントの回答（文字列）
    """
    # チャット形式モデル用の入力を整形
    model_input = {
        "messages": [{"role": "user", "content": query}]
    }
    
    # モデルで予測を実行
    response = loaded_model.predict(model_input)
    
    # 最後のメッセージ（エージェントの回答）を返す
    return response[-1]["messages"][-1]['content']

# COMMAND ----------

# ========================================
# 評価を実行
# ========================================

print("\n🚀 登録したモデルで評価を実行中...")
print("   （この処理には数分かかる場合があります）\n")

with mlflow.start_run(run_name="medical_agent_evaluation_v2_from_model") as run:
    results = mlflow.genai.evaluate(
        data=eval_data,                        # 同じ評価データセットを使用
        predict_fn=predict_wrapper_from_model,  # ロードしたモデル用のラッパー
        scorers=scorers,                       # 同じスコアラーを使用
    )
    
    reeval_run_id = run.info.run_id

print("\n✅ 評価が完了しました")
print(f"   Run ID: {reeval_run_id}")
print("\n💡 最初の評価と比較して、結果が一致していることを確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # フェーズ2: モデルの登録と管理
# MAGIC ---
# MAGIC
# MAGIC ## Unity Catalogによるモデル管理
# MAGIC
# MAGIC 評価が完了したエージェントをUnity Catalogに登録することで、
# MAGIC 企業レベルのガバナンスと管理機能を活用できます。
# MAGIC
# MAGIC ### Unity Catalog登録のメリット
# MAGIC
# MAGIC - **中央管理**: すべてのモデルを一元管理
# MAGIC - **アクセス制御**: 細かい権限設定で安全性を確保
# MAGIC - **リネージ追跡**: データやツールとの依存関係を可視化
# MAGIC - **共有と協業**: チーム間でモデルを安全に共有
# MAGIC - **監査証跡**: 変更履歴を自動的に記録

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ12: Unity Catalogへのモデル登録
# MAGIC
# MAGIC ### Unity Catalogとは
# MAGIC
# MAGIC Unity Catalogは、Databricksの統合データガバナンスソリューションです。
# MAGIC モデルをUnity Catalogに登録すると、以下のメリットがあります：
# MAGIC
# MAGIC - **中央管理**: すべてのモデルを一元管理
# MAGIC - **アクセス制御**: 細かい権限設定が可能
# MAGIC - **リネージ追跡**: モデルの依存関係を可視化
# MAGIC - **共有**: チーム間でモデルを共有
# MAGIC
# MAGIC ### モデルの完全修飾名
# MAGIC
# MAGIC Unity Catalogでは、モデル名は以下の形式で指定します：
# MAGIC `<catalog>.<schema>.<model_name>`
# MAGIC
# MAGIC 例: `exercise_john_doe.agent.medical_assistant_agent`

# COMMAND ----------

# ========================================
# Unity Catalogへの登録準備
# ========================================

# Unity CatalogをMLflowのレジストリとして設定
mlflow.set_registry_uri("databricks-uc")

# モデル名を定義
model_name = "medical_assistant_agent"

# Unity Catalog上の完全修飾名を構築
UC_MODEL_NAME = f"{catalog_name}.{system_schema_name}.{model_name}"

print(f"📝 Unity Catalogにモデルを登録中...")
print(f"   モデル名: {UC_MODEL_NAME}")

# ========================================
# モデルをUnity Catalogに登録
# ========================================

uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,  # 登録するモデルのURI
    name=UC_MODEL_NAME                      # Unity Catalog上の名前
)

print(f"\n✅ モデルをUnity Catalogに登録しました")
print(f"   バージョン: {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ13: モデルのリネージを確認
# MAGIC
# MAGIC ### リネージとは
# MAGIC
# MAGIC モデルが依存するリソース（テーブル、関数、インデックス）の
# MAGIC 関係性を可視化したものです。
# MAGIC
# MAGIC ### Unity Catalogのリネージ機能
# MAGIC
# MAGIC Unity Catalogに登録されたモデルは、以下の依存関係を
# MAGIC 自動的に追跡します：
# MAGIC
# MAGIC - 使用したテーブル（患者データなど）
# MAGIC - 使用したUnity Catalog関数（get_patient_data、predict_cancerなど）
# MAGIC - 使用したベクトル検索インデックス
# MAGIC - 使用したLLMエンドポイント
# MAGIC
# MAGIC ### リネージの確認方法
# MAGIC
# MAGIC 下のリンクをクリックして、Unity CatalogのUIで
# MAGIC 「依存関係」タブを確認してください。

# COMMAND ----------

from IPython.display import display, HTML

# DatabricksのホストURLを取得
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Unity Catalog上のモデルページへのURLを構築
model_url = f"https://{workspace_url}/explore/data/models/{UC_MODEL_NAME}"

# クリック可能なHTMLリンクを作成
html_link = f'<a href="{model_url}" target="_blank">📊 Unity Catalogでモデルの詳細を表示</a>'

display(HTML(html_link))

print("\n💡 リンク先で「依存関係」タブを確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # フェーズ3: 本番環境へのデプロイ
# MAGIC ---
# MAGIC
# MAGIC ## Model Servingによる本番化
# MAGIC
# MAGIC Unity Catalogに登録したエージェントを、REST APIとして公開します。
# MAGIC これにより、外部アプリケーションから呼び出せるようになります。
# MAGIC
# MAGIC ### Model Servingの特徴
# MAGIC
# MAGIC - **スケーラビリティ**: 負荷に応じて自動的にスケール
# MAGIC - **高可用性**: 99.9%のSLA保証
# MAGIC - **低レイテンシ**: 高速なレスポンス
# MAGIC - **監視機能**: リアルタイムでパフォーマンスを追跡
# MAGIC - **A/Bテスト**: 複数のモデルバージョンを並行運用
# MAGIC
# MAGIC ### レビューアプリ
# MAGIC
# MAGIC デプロイと同時に、チームメンバーがブラウザで簡単にテストできる
# MAGIC レビューアプリも自動的に作成されます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ14: エージェントのデプロイ
# MAGIC
# MAGIC ### Model Servingとは
# MAGIC
# MAGIC Model Servingは、機械学習モデルをREST APIとして公開する
# MAGIC Databricksのサービスです。
# MAGIC
# MAGIC ### デプロイのメリット
# MAGIC
# MAGIC - **本番環境での使用**: 実際のアプリケーションから呼び出せる
# MAGIC - **スケーラビリティ**: 負荷に応じて自動的にスケール
# MAGIC - **監視**: リアルタイムでパフォーマンスを監視
# MAGIC - **A/Bテスト**: 複数のモデルバージョンを比較
# MAGIC
# MAGIC ### 環境変数の設定
# MAGIC
# MAGIC デプロイ時に、エージェントが使用する環境変数を設定します。
# MAGIC これにより、デプロイ後も正しいリソースにアクセスできます。
# MAGIC
# MAGIC ### agents.deploy() とは
# MAGIC
# MAGIC Databricksが提供する高レベルなデプロイAPI。
# MAGIC 以下の処理を自動的に実行します：
# MAGIC
# MAGIC 1. **レビューアプリの作成**: チーム内でテストできるアプリ
# MAGIC 2. **Model Servingエンドポイントの作成**: 本番用API
# MAGIC 3. **環境変数の設定**: エージェントが必要とする設定
# MAGIC 4. **ヘルスチェック**: デプロイが成功したか確認

# COMMAND ----------

from databricks import agents

# ========================================
# デプロイ用の環境変数を設定
# ========================================

environment_vars = {
    "LLM_ENDPOINT_NAME": os.environ["LLM_ENDPOINT_NAME"],
    "VS_NAME": os.environ["VS_NAME"],
    "UC_TOOL_NAMES": os.environ["UC_TOOL_NAMES"],
}

print("🚀 エージェントをデプロイ中...")
print(f"   モデル: {UC_MODEL_NAME}")
print(f"   バージョン: {uc_registered_model_info.version}")
print("\n   （この処理には10〜15分かかる場合があります）\n")

# ========================================
# エージェントをデプロイ
# ========================================

deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME,                          # デプロイするモデルの名前
    model_version=uc_registered_model_info.version,    # デプロイするモデルのバージョン
    tags={"endpointSource": "Medical Agent Lab"},      # タグ（管理用）
    environment_vars=environment_vars,                 # 環境変数
    timeout=900,                                       # タイムアウト（秒）: 15分
)

print("\n✅ デプロイが完了しました")
print(f"   エンドポイント名: {deployment_info.endpoint_name}")
print(f"   レビューアプリURL: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ15: デプロイしたエージェントのテスト
# MAGIC
# MAGIC デプロイが成功したら、実際にAPIを呼び出してテストします。

# COMMAND ----------

import requests
import json

# ========================================
# エンドポイントのURLを構築
# ========================================

# DatabricksのホストURLを取得
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Model ServingエンドポイントのURL
endpoint_url = f"https://{workspace_url}/serving-endpoints/{deployment_info.endpoint_name}/invocations"

# ========================================
# テストリクエストを送信
# ========================================

test_query = "乳がんの早期発見にはどのような検査がありますか？"

# リクエストボディを構築
request_body = {
    "messages": [{
        "role": "user",
        "content": test_query
    }]
}

print(f"🧪 デプロイしたエンドポイントをテスト中...")
print(f"   質問: {test_query}\n")

# 認証トークンを取得
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
token = w.config.token

# HTTPリクエストを送信
response = requests.post(
    endpoint_url,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json=request_body,
    timeout=60
)

# レスポンスを確認
if response.status_code == 200:
    result = response.json()
    answer = result['messages'][-1]['content']
    print(f"✅ 回答:\n{answer}")
else:
    print(f"❌ エラー: {response.status_code}")
    print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ：エージェント評価・登録・デプロイの振り返り
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC このハンズオンで、AIエージェントを本番環境で使えるようにする
# MAGIC 完全なプロセスを習得しました：
# MAGIC
# MAGIC #### フェーズ1: エージェントの評価
# MAGIC - ✅ 評価データセットの作成方法
# MAGIC - ✅ MLflowを使った自動評価の実装
# MAGIC - ✅ LLMジャッジによるスコアリング
# MAGIC - ✅ Correctness（正確性）とRetrievalSufficiency（検索十分性）の測定
# MAGIC - ✅ 評価結果の分析と改善点の特定
# MAGIC
# MAGIC #### フェーズ2: モデルの登録と管理
# MAGIC - ✅ MLflowモデルとしての登録方法
# MAGIC - ✅ Unity Catalogでの一元管理
# MAGIC - ✅ モデルのバージョン管理
# MAGIC - ✅ リネージによる依存関係の可視化
# MAGIC - ✅ アクセス制御とガバナンス
# MAGIC
# MAGIC #### フェーズ3: 本番環境へのデプロイ
# MAGIC - ✅ Model Servingエンドポイントの作成
# MAGIC - ✅ 環境変数の適切な設定
# MAGIC - ✅ レビューアプリの活用方法
# MAGIC - ✅ REST APIを通じたエージェントの呼び出し
# MAGIC - ✅ デプロイ後のテストと検証
# MAGIC
# MAGIC ### AIエージェント開発の全体像
# MAGIC
# MAGIC 今回のハンズオンシリーズ（01〜05）で、以下の完全なライフサイクルを学びました：
# MAGIC
# MAGIC 1. **01. データ準備**: PDFからベクトル検索用データを作成
# MAGIC 2. **02. ツール準備**: Unity Catalog関数の作成
# MAGIC 3. **03. ReACT理解**: ReACTエージェントの基本動作を理解
# MAGIC 4. **04. エージェント開発**: LangGraphを使った実用的なエージェント構築
# MAGIC 5. **05. 評価・登録・デプロイ**: 本番環境への展開 ← 今ここ
# MAGIC
# MAGIC ### 本番運用に向けて
# MAGIC
# MAGIC デプロイが完了したら、以下の点に注意して運用してください：
# MAGIC
# MAGIC #### 継続的な監視
# MAGIC - エンドポイントのレスポンスタイムを監視
# MAGIC - エラー率をトラッキング
# MAGIC - 使用量とコストを管理
# MAGIC - ユーザーフィードバックを収集
# MAGIC
# MAGIC #### 継続的な改善
# MAGIC - 定期的な再評価の実施
# MAGIC - 評価データセットの拡充
# MAGIC - システムプロンプトの最適化
# MAGIC - 新しいツールの追加
# MAGIC
# MAGIC #### バージョン管理
# MAGIC - 改善版を新しいバージョンとして登録
# MAGIC - A/Bテストで性能を比較
# MAGIC - リグレッション防止のための評価
# MAGIC - ロールバック計画の準備
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC エージェントをさらに改善し、本格的に運用するには：
# MAGIC
# MAGIC #### 1. 評価の強化
# MAGIC - 評価データセットの拡充（より多様なシナリオ）
# MAGIC - 追加スコアラーの有効化（Safety、RelevanceToQueryなど）
# MAGIC - カスタムスコアラーの作成（業務固有の評価基準）
# MAGIC - [合成データ生成](https://www.databricks.com/jp/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities)で大規模評価
# MAGIC
# MAGIC #### 2. 監視とモニタリング
# MAGIC - [MLflowモニタリング](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/)の設定
# MAGIC - ダッシュボードでの可視化
# MAGIC - アラート設定（エラー率、レスポンスタイム）
# MAGIC - ユーザーフィードバック収集システムの構築
# MAGIC
# MAGIC #### 3. システムプロンプトの最適化
# MAGIC - 評価結果を基に改善
# MAGIC - A/Bテストで効果を検証
# MAGIC - Few-shot例の追加
# MAGIC - Chain-of-Thought プロンプティングの導入
# MAGIC
# MAGIC #### 4. アプリケーション統合
# MAGIC - [Databricks Apps](https://docs.databricks.com/aws/ja/dev-tools/databricks-apps/get-started)でチャットUIを構築
# MAGIC - 既存の業務アプリケーションに組み込み
# MAGIC - Slackやチャットツールとの連携
# MAGIC - モバイルアプリからの呼び出し
# MAGIC
# MAGIC #### 5. スケーリングと最適化
# MAGIC - エンドポイントのスケーリング設定の調整
# MAGIC - キャッシュ戦略の導入
# MAGIC - コスト最適化（モデルサイズ、インスタンスタイプ）
# MAGIC - マルチリージョンデプロイ
# MAGIC
# MAGIC #### 6. ガバナンスとセキュリティ
# MAGIC - アクセス制御の強化
# MAGIC - データプライバシーの確保
# MAGIC - 監査ログの活用
# MAGIC - コンプライアンス要件への対応

# COMMAND ----------

# MAGIC %md
# MAGIC ## 参考リソース
# MAGIC
# MAGIC ### Databricks公式ドキュメント
# MAGIC
# MAGIC - [エージェント評価](https://docs.databricks.com/aws/ja/generative-ai/agent-evaluation)
# MAGIC - [MLflow - コードからのモデル](https://mlflow.org/docs/latest/models.html#models-from-code)
# MAGIC - [MLflowのスコアラー](https://docs.databricks.com/gcp/ja/mlflow3/genai/eval-monitor/custom-judge/meets-guidelines)
# MAGIC - [合成データ生成](https://www.databricks.com/jp/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities)
# MAGIC - [MLflow評価と監視](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/)
# MAGIC - [Databricks Apps](https://docs.databricks.com/aws/ja/dev-tools/databricks-apps/get-started)
# MAGIC
# MAGIC ### おわりに
# MAGIC
# MAGIC お疲れ様でした！このハンズオンシリーズ（01〜04）を通じて、
# MAGIC Databricks上で実用的なAIエージェントを構築、評価、デプロイする
# MAGIC 一連の流れを学びました。
# MAGIC
# MAGIC ぜひ、学んだ知識を活用して、あなた自身のエージェントを
# MAGIC 構築してみてください！
