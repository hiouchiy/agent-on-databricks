# Databricks notebook source
# MAGIC %md
# MAGIC # AIエージェント評価ハンズオンラボ
# MAGIC
# MAGIC ## このノートブックで学ぶこと
# MAGIC
# MAGIC このハンズオンでは、前のノートブック（03.agent_develop）で構築した
# MAGIC 医療アシスタントエージェントの性能を評価する方法を学びます。
# MAGIC
# MAGIC ### エージェント評価とは？
# MAGIC
# MAGIC エージェントが期待通りに動作しているかを客観的に測定する作業です。
# MAGIC 以下のような質問に答えることができます：
# MAGIC
# MAGIC - エージェントは正確な情報を提供しているか？
# MAGIC - 検索した情報を適切に活用しているか？
# MAGIC - 回答は質問に対して適切か？
# MAGIC - 安全性やバイアスの問題はないか？
# MAGIC
# MAGIC ### このノートブックの流れ
# MAGIC
# MAGIC 1. **環境準備**: 必要なライブラリのインストール
# MAGIC 2. **エージェントの読み込み**: simple_agent.pyからエージェントを読み込む
# MAGIC 3. **評価データセットの作成**: テスト用の質問と期待される回答を準備
# MAGIC 4. **評価の実行**: MLflowを使ってエージェントを評価
# MAGIC 5. **結果の分析**: 評価結果を確認し、改善点を特定
# MAGIC 6. **モデルの登録とデプロイ**: Unity Catalogへの登録と本番環境へのデプロイ
# MAGIC
# MAGIC ### simple_agent.pyファイルについて
# MAGIC
# MAGIC このノートブックは、別ファイル（`simple_agent.py`）に定義されたエージェントを
# MAGIC 読み込んで評価します。simple_agent.pyには以下が含まれます：
# MAGIC
# MAGIC - LLMの設定
# MAGIC - ベクトル検索ツールの設定
# MAGIC - Unity Catalog関数ツールの設定
# MAGIC - システムプロンプト
# MAGIC - エージェントの構築ロジック
# MAGIC
# MAGIC この構成により、エージェントのコードと評価コードを分離でき、
# MAGIC 管理とバージョン管理が容易になります。

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
# MAGIC ## ステップ7: 評価用のラッパー関数を定義
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
# MAGIC ## ステップ8: 評価データセットの作成
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
# MAGIC ## ステップ9: 評価指標（スコアラー）の定義
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
# MAGIC ## ステップ10: エージェントの評価を実行
# MAGIC
# MAGIC ### mlflow.genai.evaluate() とは
# MAGIC
# MAGIC MLflowが提供するエージェント評価フレームワークです。
# MAGIC 以下の処理を自動的に実行します：
# MAGIC
# MAGIC 1. **評価データセットの各質問をエージェントに送信**
# MAGIC    - predict_wrapper関数を使用
# MAGIC
# MAGIC 2. **回答を期待値と比較**
# MAGIC    - 定義したスコアラーを使用
# MAGIC
# MAGIC 3. **スコアを計算**
# MAGIC    - 各スコアラーが0〜1のスコアを出力
# MAGIC
# MAGIC 4. **結果をMLflowに記録**
# MAGIC    - スコア、入力、出力、トレース情報をすべて記録
# MAGIC
# MAGIC ### MLflow Run
# MAGIC
# MAGIC 評価は、MLflowのRunとして記録されます。
# MAGIC Run内には以下の情報が含まれます：
# MAGIC
# MAGIC - 評価スコア（メトリクス）
# MAGIC - 各テストケースの詳細
# MAGIC - エージェントのトレース情報
# MAGIC - 使用したツールの履歴
# MAGIC
# MAGIC ### 評価結果の確認方法
# MAGIC
# MAGIC 評価完了後、以下の方法で結果を確認できます：
# MAGIC
# MAGIC 1. **MLflow UI**: Databricksの「実験」タブから確認
# MAGIC 2. **results変数**: このセルで返される結果オブジェクト
# MAGIC 3. **自動生成されたレポート**: 視覚的なダッシュボード

# COMMAND ----------

print("🚀 評価を実行中...")
print("   （この処理には数分かかる場合があります）\n")

# MLflow Runを開始
with mlflow.start_run(run_name="medical_agent_evaluation_v1") as run:
    # エージェントの評価を実行
    results = mlflow.genai.evaluate(
        data=eval_data,              # 評価データセット
        predict_fn=predict_wrapper,  # 予測関数
        scorers=scorers,             # スコアラーのリスト
    )
    
    # Run IDを保存（後で参照するため）
    run_id = run.info.run_id

print("\n✅ 評価が完了しました")
print(f"   Run ID: {run_id}")
print("\n💡 MLflow UIで詳細な評価結果を確認できます")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ11: 評価結果の確認
# MAGIC
# MAGIC 評価結果を表示して、エージェントの性能を確認します。

# COMMAND ----------

# 評価結果のサマリーを表示
print("📊 評価結果サマリー:")
print("="*50)

# メトリクスを表示
if hasattr(results, 'metrics'):
    for metric_name, metric_value in results.metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
else:
    print("   メトリクス情報が利用できません")

print("\n💡 詳細な評価結果はMLflow UIで確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ12: エージェントをMLflowモデルとして登録
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
# MAGIC ### agent.pyファイルの登録
# MAGIC
# MAGIC `python_model="agent.py"` を指定することで、
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
# MAGIC ## ステップ13: 登録したモデルで再評価（オプション）
# MAGIC
# MAGIC ### なぜ再評価するのか
# MAGIC
# MAGIC 登録したMLflowモデルが正しく動作するか確認するため、
# MAGIC 再度評価を実行します。
# MAGIC
# MAGIC ### 最初の評価との違い
# MAGIC
# MAGIC - **最初の評価**: agent.pyから直接読み込んだAGENTオブジェクトを評価
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
        "messages": [{
            "role": "user", 
            "content": query
        }]
    }
    
    # モデルで予測を実行
    response = loaded_model.predict(model_input)
    
    # 最後のメッセージ（エージェントの回答）を返す
    return response[-1]["messages"][-1]['content']

# COMMAND ----------

predict_wrapper_from_model("こんにちは")

# COMMAND ----------

# ========================================
# 再評価を実行
# ========================================

print("\n🚀 登録したモデルで再評価を実行中...")
print("   （この処理には数分かかる場合があります）\n")

with mlflow.start_run(run_name="medical_agent_evaluation_v2_from_model") as run:
    results = mlflow.genai.evaluate(
        data=eval_data,                        # 同じ評価データセットを使用
        predict_fn=predict_wrapper_from_model,  # ロードしたモデル用のラッパー
        scorers=scorers,                       # 同じスコアラーを使用
    )
    
    reeval_run_id = run.info.run_id

print("\n✅ 再評価が完了しました")
print(f"   Run ID: {reeval_run_id}")
print("\n💡 最初の評価と比較して、結果が一致していることを確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ14: Unity Catalogへのモデル登録
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
# MAGIC ## ステップ15: モデルのリネージを確認
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
# MAGIC ## ステップ16: エージェントのデプロイ
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
# MAGIC ## ステップ17: デプロイしたエージェントのテスト
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
# MAGIC ## まとめ：エージェント評価とデプロイの振り返り
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC このハンズオンで、以下のスキルを習得しました：
# MAGIC
# MAGIC #### 1. エージェントの評価
# MAGIC - 評価データセットの作成方法
# MAGIC - MLflowを使った自動評価
# MAGIC - LLMジャッジによるスコアリング
# MAGIC - 評価結果の分析と改善
# MAGIC
# MAGIC #### 2. モデル管理
# MAGIC - MLflowへのモデル登録
# MAGIC - Unity Catalogでの一元管理
# MAGIC - モデルのバージョン管理
# MAGIC - リネージの追跡
# MAGIC
# MAGIC #### 3. 本番環境へのデプロイ
# MAGIC - Model Servingエンドポイントの作成
# MAGIC - 環境変数の設定
# MAGIC - レビューアプリの活用
# MAGIC - デプロイしたエージェントのテスト
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC エージェントをさらに改善するには：
# MAGIC
# MAGIC 1. **評価データセットの拡充**
# MAGIC    - より多様なテストケースを追加
# MAGIC    - エッジケースのテスト
# MAGIC    - 実際のユーザーフィードバックを反映
# MAGIC
# MAGIC 2. **スコアラーの調整**
# MAGIC    - 追加のスコアラーを有効化
# MAGIC    - カスタムスコアラーの作成
# MAGIC    - スコアの閾値を設定
# MAGIC
# MAGIC 3. **システムプロンプトの最適化**
# MAGIC    - 評価結果を基に改善
# MAGIC    - A/Bテストで比較
# MAGIC
# MAGIC 4. **継続的な監視と改善**
# MAGIC    - [MLflowモニタリング](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/)の設定
# MAGIC    - 定期的な再評価の実施
# MAGIC    - ユーザーフィードバックの収集
# MAGIC
# MAGIC 5. **アプリケーション統合**
# MAGIC    - [Databricks Apps](https://docs.databricks.com/aws/ja/dev-tools/databricks-apps/get-started)との連携
# MAGIC    - 外部アプリケーションからの呼び出し
# MAGIC    - チャットボットUIの構築

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
