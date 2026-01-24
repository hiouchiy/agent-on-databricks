# Databricks notebook source
# MAGIC %md
# MAGIC # ReACTエージェントの仕組みを理解する
# MAGIC
# MAGIC ## ReACT (Reasoning + Acting) とは
# MAGIC
# MAGIC ReACTは、LLMが以下のサイクルを繰り返して問題を解決する手法です：
# MAGIC
# MAGIC 1. **Thought (思考)**: 「次に何をすべきか」をLLMが考える
# MAGIC 2. **Action (行動)**: ツールを実行する、または最終回答を返す
# MAGIC 3. **Observation (観察)**: ツールの実行結果を確認する
# MAGIC
# MAGIC このノートブックでは、医療マニュアル検索を例に、ReACTの動作を**1ステップずつ**確認していきます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境準備

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk openai databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: カタログとスキーマの設定

# COMMAND ----------

# 現在のユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

CATALOG = f"exercise_{clean_username}"
SCHEMA = "agent"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"✅ 環境セットアップ完了")
print(f"   カタログ: {CATALOG}")
print(f"   スキーマ: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: LLMモデルの選択

# COMMAND ----------

import random
MODEL_NAME = random.choice(["databricks-llama-4-maverick", "databricks-qwen3-next-80b-a3b-instruct"])
print(MODEL_NAME + " が選択されました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: OpenAIクライアントの初期化

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI
import json

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()
print("✅ クライアントの初期化が完了しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: ツールの定義
# MAGIC
# MAGIC ReACTでは、LLMが使えるツール（関数）を定義します。
# MAGIC ここでは、医療マニュアルを検索するツールを1つだけ定義します。
# MAGIC
# MAGIC LLMはこのツールの説明（description）を読んで、「使うべきかどうか」を判断します。

# COMMAND ----------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_medical_manual",
            "description": "最新の医療マニュアルを検索します。医療や治療方法に関する質問に答えるために使用してください。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The string used to query the index with and identify the most similar vectors and return the associated documents."
                    },
                },
                "required": ["query"]
            }
        }
    }
]

print("【定義したツール】")
print(json.dumps(tools, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔄 ReACTサイクル 第1ラウンド
# MAGIC ---
# MAGIC
# MAGIC ## 💭 THOUGHT (思考): LLMに「次に何をすべきか」考えさせる
# MAGIC
# MAGIC ユーザーの質問「胃がんの治療法を箇条書きで簡潔にまとめてください」に対して、
# MAGIC LLMが次のどちらかを選択します：
# MAGIC
# MAGIC - **A) ツールを使う**: `search_medical_manual` で検索が必要だと判断
# MAGIC - **B) 直接回答**: 既存の知識で回答できると判断
# MAGIC
# MAGIC 実際に何を選ぶか見てみましょう。

# COMMAND ----------

messages = [
    {"role": "system", "content": "あなたは優秀な医療アシスタントです。がん（Cancer）に関する質問には必ず最新の医療マニュアルを検索し、その内容のみに基づいて回答してください。それ以外の医療系質問にはあなたの持っている知識の範囲で回答してください。それ以外の質問には回答せず、医療系アシスタントであることをお知らせください。"},
    {"role": "user", "content": "胃がんの治療法を箇条書きで簡潔にまとめてください。"}
]

# LLMに次のアクションを考えさせる
response1 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # LLMに判断を委ねる
)

message1 = response1.choices[0].message
print("【LLMの返答データ（JSON形式）】")
print(message1.to_json())
print("\n" + "="*80)

# LLMの判断を確認
if getattr(message1, "tool_calls", None):
    tc = message1.tool_calls[0]
    args = json.loads(tc.function.arguments or "{}")
    print(f"✅ LLMの判断: ツール「{tc.function.name}」を使用します")
    print(f"   検索クエリ: {args}")
else:
    print("✅ LLMの判断: ツールを使わずに直接回答します")
    print(f"   回答内容: {message1.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 ACTION (行動): ツールを実際に実行する
# MAGIC
# MAGIC LLMが「search_medical_manualを使う」と判断したので、
# MAGIC 実際にベクトル検索を実行します。
# MAGIC
# MAGIC ここでは、LLMが生成した検索クエリを使って、
# MAGIC 医療マニュアルのベクトルインデックスから関連情報を取得します。

# COMMAND ----------

# LLMが生成した検索クエリを取得
query_text = args["query"]
print(f"🔍 検索クエリ: {query_text}\n")

from databricks.vector_search.client import VectorSearchClient

# ベクトル検索を実行
vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="vs_endpoint", 
    index_name=f"{CATALOG}.{SCHEMA}.manuals_index"
)

results = index.similarity_search(
    query_text=query_text,
    columns=["pdf_path", "page_number", "chunk_text"],
    num_results=3  # 上位3件を取得
)

# 検索結果を整形
all_results = ""
for i, result in enumerate(results.get("result", {}).get("data_array", []), 1):
    all_results += f"\n--- 結果 {i} ---"
    all_results += f"\nPDFファイル: {result[0]}"
    all_results += f"\nページ番号: {result[1]}"
    all_results += f"\nテキスト: {result[2]}"
    all_results += "\n"

print("【検索結果】")
print(all_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👁️ OBSERVATION (観察): 検索結果をLLMに渡す
# MAGIC
# MAGIC これがReACTの重要なポイントです：
# MAGIC
# MAGIC 1. LLMがツールを呼び出す（Action）
# MAGIC 2. ツールが実行される
# MAGIC 3. **実行結果をLLMに返す（Observation）** ← 今ここ
# MAGIC 4. LLMが結果を見て、次のアクションを考える
# MAGIC
# MAGIC 検索結果を会話履歴に追加して、LLMに「観察」させます。

# COMMAND ----------

# 会話履歴を更新
messages = [
    {"role": "system", "content": "あなたは優秀な医療アシスタントです。がん（Cancer）に関する質問には必ず最新の医療マニュアルを検索し、その内容のみに基づいて回答してください。それ以外の医療系質問にはあなたの持っている知識の範囲で回答してください。それ以外の質問には回答せず、医療系アシスタントであることをお知らせください。"},
    {"role": "user", "content": "胃がんの治療法を箇条書きで簡潔にまとめてください。"},
    message1.to_dict(),  # 第1ラウンドのLLM応答（ツール呼び出し）
    {"role": "tool", "content": all_results, "tool_call_id": message1.tool_calls[0].id},  # ツールの実行結果
]

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔄 ReACTサイクル 第2ラウンド
# MAGIC ---
# MAGIC
# MAGIC ## 💭 THOUGHT (思考): 検索結果を基に次のアクションを考える
# MAGIC
# MAGIC LLMは検索結果を受け取り、再び次のアクションを考えます：
# MAGIC
# MAGIC - **A) さらにツールを使う**: 情報が不足していれば再検索
# MAGIC - **B) 最終回答を生成**: 十分な情報が揃っていれば回答を作成
# MAGIC
# MAGIC 会話履歴に「ツール実行結果」を追加してから、再度LLMを呼び出します。

# COMMAND ----------

# 再度LLMに考えさせる
response2 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

message2 = response2.choices[0].message
print("【LLMの返答データ（JSON形式）】")
print(message2.to_json())
print("\n" + "="*80)

# LLMの判断を確認
if getattr(message2, "tool_calls", None):
    tc = message2.tool_calls[0]
    args = json.loads(tc.function.arguments or "{}")
    print(f"✅ LLMの判断: さらにツール「{tc.function.name}」を使用します")
    print(f"   検索クエリ: {args}")
else:
    print("✅ LLMの判断: 十分な情報が揃ったので最終回答を生成します")
    print("\n" + "="*80)
    print("【最終回答】")
    print("="*80)
    print(message2.content)
    print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ ReACTサイクルの完了
# MAGIC ---
# MAGIC
# MAGIC ## 振り返り
# MAGIC
# MAGIC 今回の実行では、以下のようなReACTサイクルが発生しました：
# MAGIC
# MAGIC ### 第1ラウンド
# MAGIC 1. **💭 Thought**: 「胃がんの治療法」について聞かれた → 医療マニュアルを検索すべき
# MAGIC 2. **🔧 Action**: `search_medical_manual` ツールを実行
# MAGIC 3. **👁️ Observation**: 検索結果を取得
# MAGIC
# MAGIC ### 第2ラウンド
# MAGIC 1. **💭 Thought**: 検索結果に十分な情報がある → 最終回答を生成できる
# MAGIC 2. **🎯 Final Answer**: 検索結果を基に箇条書きで回答を作成
# MAGIC
# MAGIC ## ReACTの利点
# MAGIC
# MAGIC - ✅ **最新情報にアクセス**: LLMの訓練データにない情報も検索できる
# MAGIC - ✅ **情報源が明確**: どのPDFの何ページから情報を取得したか分かる
# MAGIC - ✅ **段階的な問題解決**: 必要に応じて複数回ツールを呼び出せる
# MAGIC - ✅ **透明性**: 各ステップで何が起きているか確認できる
# MAGIC
# MAGIC ## 次のステップ
# MAGIC
# MAGIC このノートブックでは、ReACTの基本的な動作を手動で確認しました。
# MAGIC 次は **LangGraph** を使って、このサイクルを自動化する方法を学びます。
