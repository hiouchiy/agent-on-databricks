# Databricks notebook source
# MAGIC %md
# MAGIC # AIエージェント構築ハンズオンラボ
# MAGIC
# MAGIC ## このノートブックで学ぶこと
# MAGIC
# MAGIC このハンズオンでは、Databricks上でLangChainとLangGraphを使って、
# MAGIC 実用的なAIエージェント（自律的に動作するAIアシスタント）を構築し、
# MAGIC その性能を客観的に評価する方法を学びます。
# MAGIC
# MAGIC ### AIエージェントとは？
# MAGIC
# MAGIC AIエージェントは、単に質問に答えるだけでなく、以下のことができます：
# MAGIC
# MAGIC - **自分で考えて行動する**: 質問に答えるために何をすべきか判断
# MAGIC - **ツールを使う**: データベース検索やAPI呼び出しなどを実行
# MAGIC - **複数ステップの処理**: 必要に応じて複数の行動を組み合わせる
# MAGIC
# MAGIC ### 構築するエージェントの機能
# MAGIC
# MAGIC 今回作るのは「医療アシスタントエージェント」です：
# MAGIC
# MAGIC 1. **質問の理解**: ユーザーの質問内容を理解
# MAGIC 2. **マニュアル検索**: ベクトル検索で医療マニュアルから関連情報を取得
# MAGIC 3. **患者データ取得**: Unity Catalog関数で患者の特徴量データを取得
# MAGIC 4. **乳がん予測**: 機械学習モデルで乳がんの悪性/良性を予測
# MAGIC 5. **回答生成**: 検索結果や予測結果を基に分かりやすく回答
# MAGIC 6. **適切な対応**: 医療外の質問には丁寧に断る
# MAGIC
# MAGIC ### エージェントの評価
# MAGIC
# MAGIC 構築したエージェントの品質を確保するために、以下の評価を行います：
# MAGIC
# MAGIC 1. **評価データセットの作成**: 様々なシナリオのテストケースを準備
# MAGIC 2. **自動評価**: MLflowの評価フレームワークで複数の指標を計測
# MAGIC 3. **LLMジャッジ**: 別のLLMを使って回答の品質を客観的に評価
# MAGIC 4. **結果の分析**: スコアを確認し、改善ポイントを特定
# MAGIC
# MAGIC これにより、エージェントが本番環境で使えるレベルに達しているか判断できます。
# MAGIC
# MAGIC ### ハンズオンの流れ
# MAGIC
# MAGIC #### フェーズ1: エージェントの構築
# MAGIC 1. 環境準備（ライブラリのインストール）
# MAGIC 2. MLflow自動ロギングの有効化
# MAGIC 3. LLM（大規模言語モデル）の設定
# MAGIC 4. ツールの設定（ベクトル検索 + Unity Catalog関数）
# MAGIC 5. エージェントの構築
# MAGIC 6. 実際に動かして確認
# MAGIC
# MAGIC #### フェーズ2: エージェントの評価
# MAGIC 7. 評価用ラッパー関数の定義
# MAGIC 8. 評価データセットの作成
# MAGIC 9. 評価指標（スコアラー）の設定
# MAGIC 10. 自動評価の実行
# MAGIC 11. 評価結果の分析

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境準備
# MAGIC
# MAGIC ### 必要なライブラリのインストール
# MAGIC
# MAGIC 以下のライブラリをインストールします：
# MAGIC
# MAGIC - **unitycatalog-ai[databricks]**: Unity CatalogとAIの統合パッケージ
# MAGIC - **unitycatalog-langchain[databricks]**: Unity CatalogとLangChainの連携用
# MAGIC - **databricks-langchain**: DatabricksとLangChainの連携用
# MAGIC
# MAGIC ### 注意事項
# MAGIC
# MAGIC インストール後、Pythonを再起動します。
# MAGIC これにより、新しくインストールしたライブラリが正しく読み込まれます。

# COMMAND ----------

# Unity Catalog AI統合パッケージのインストール
%pip install unitycatalog-ai[databricks]
%pip install unitycatalog-langchain[databricks]

# Databricks LangChain統合パッケージのインストール
%pip install databricks-langchain databricks-agents uv

# Pythonを再起動して新しいライブラリを読み込む
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: MLflow自動ロギングの有効化
# MAGIC
# MAGIC ### MLflow自動ロギングとは
# MAGIC
# MAGIC MLflowの自動ロギング機能を有効にすると、以下の情報が自動的に記録されます：
# MAGIC
# MAGIC - **入力と出力**: エージェントへの質問と回答
# MAGIC - **使用したツール**: どのツールをいつ使ったか
# MAGIC - **処理時間**: 各ステップにかかった時間
# MAGIC - **エラー情報**: 問題が発生した場合の詳細
# MAGIC
# MAGIC ### なぜ重要なのか
# MAGIC
# MAGIC - デバッグが簡単になる
# MAGIC - パフォーマンスの分析ができる
# MAGIC - 本番環境での動作を監視できる

# COMMAND ----------

import mlflow

# LangChainの自動ロギングを有効化
mlflow.langchain.autolog()

print("✓ MLflow自動ロギングが有効になりました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: LLM（大規模言語モデル）の設定
# MAGIC
# MAGIC ### LLMとは
# MAGIC
# MAGIC LLM（Large Language Model）は、人間のように文章を理解し、生成できるAIモデルです。
# MAGIC ChatGPTやClaude、Llama などが代表例です。
# MAGIC
# MAGIC ### Databricks上のLLMを使う利点
# MAGIC
# MAGIC - **セキュリティ**: データが外部に出ない
# MAGIC - **コスト管理**: 使用量を細かく管理できる
# MAGIC - **カスタマイズ**: 自社データでファインチューニング可能
# MAGIC
# MAGIC ### ChatDatabricksクラス
# MAGIC
# MAGIC LangChainの`ChatDatabricks`クラスを使うと、
# MAGIC Databricks上のLLMを簡単に呼び出せます。

# COMMAND ----------

from databricks_langchain import ChatDatabricks
import random

# 使用するLLMのエンドポイント名
# ※環境に合わせて変更してください
LLM_ENDPOINT_NAME = random.choice(["databricks-gpt-oss-20b", "databricks-llama-4-maverick", "databricks-qwen3-next-80b-a3b-instruct"])
print(LLM_ENDPOINT_NAME + " が選択されました。")

# LLMモデルを初期化
model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

print(f"✓ LLMモデルを初期化しました: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: LLMの動作確認
# MAGIC
# MAGIC 簡単な質問でLLMが正しく動作するか確認します。

# COMMAND ----------

# テスト質問を実行
test_question = "あなたについて簡潔に説明してください。"
response = model.invoke(test_question)

print(f"質問: {test_question}")
print(f"回答: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: 環境設定（カタログとスキーマ）
# MAGIC
# MAGIC エージェントが使用するツールにアクセスするための環境を設定します。
# MAGIC
# MAGIC ### 設定内容
# MAGIC
# MAGIC - **CATALOG**: ユーザーごとに一意のカタログ名を生成
# MAGIC - **SCHEMA**: エージェント用のスキーマ名を設定
# MAGIC
# MAGIC これらは前のノートブック（01.Data-Prep、02.Tool-Prep）で
# MAGIC 作成したリソースにアクセスするために必要です。

# COMMAND ----------

from databricks_langchain import VectorSearchRetrieverTool, UCFunctionToolkit

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

# ベクトル検索インデックスの完全修飾名を設定
CATALOG_NAME = CATALOG
SCHEMA_NAME = SCHEMA

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: ツールの設定
# MAGIC
# MAGIC エージェントが使用する3つのツールを設定します。
# MAGIC
# MAGIC ### ツール1: ベクトル検索ツール
# MAGIC
# MAGIC **ベクトル検索とは**:
# MAGIC テキストの「意味」を理解して類似した情報を見つける技術です。
# MAGIC
# MAGIC **従来のキーワード検索との違い**:
# MAGIC - キーワード検索: 「乳がん」で検索 → 「乳がん」という単語を含む文書のみ
# MAGIC - ベクトル検索: 「乳がん」で検索 → 「乳癌」「breast cancer」「乳腺腫瘍」なども検索
# MAGIC
# MAGIC **パラメータの説明**:
# MAGIC - `index_name`: 検索対象のインデックス名（01.Data-Prepで作成）
# MAGIC - `tool_name`: ツールの識別名（エージェントがツールを選ぶときに使用）
# MAGIC - `num_results`: 検索結果として取得する文書の数
# MAGIC - `tool_description`: ツールの説明（エージェントがいつ使うべきか判断する材料）
# MAGIC
# MAGIC ### ツール2・3: Unity Catalog関数
# MAGIC
# MAGIC **get_patient_data**:
# MAGIC - 患者IDから乳がん特徴量（30次元）を取得
# MAGIC - 02.Tool-Prepで作成したSQL関数
# MAGIC
# MAGIC **predict_cancer**:
# MAGIC - 特徴量から乳がんの悪性/良性を予測
# MAGIC - 02.Tool-Prepで作成したPython関数
# MAGIC - Databricks Model Servingエンドポイントを呼び出す

# COMMAND ----------

# ========================================
# ツール1: ベクトル検索ツール
# ========================================

# ベクトル検索インデックスの完全修飾名
VECTOR_SEARCH_INDEX = f"{CATALOG_NAME}.{SCHEMA_NAME}.manuals_index"

# ベクトル検索ツールを作成
vector_search_tool = VectorSearchRetrieverTool(
    index_name=VECTOR_SEARCH_INDEX,
    tool_name="search_medical_manual",  # ツールの識別名
    num_results=10,  # 上位10件の関連文書を取得
    tool_description="最新の医療マニュアルを検索します。医療や治療方法に関する質問に答えるために使用してください。",
    disable_notice=True  # 通知メッセージを無効化
)

print(f"✓ ベクトル検索ツールを作成しました")
print(f"   インデックス: {VECTOR_SEARCH_INDEX}")
print(f"   ツール名: search_medical_manual")

# ========================================
# ツール2・3: Unity Catalog関数ツール
# ========================================

# Unity Catalog関数の完全修飾名のリスト
UC_TOOL_NAMES = [
    f"{CATALOG_NAME}.{SCHEMA_NAME}.get_patient_data",   # 患者データ取得関数
    f"{CATALOG_NAME}.{SCHEMA_NAME}.predict_cancer"      # 乳がん予測関数
]

# Unity Catalog関数をツールとして登録
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)

print(f"\n✓ Unity Catalog関数ツールを作成しました")
for tool_name in UC_TOOL_NAMES:
    print(f"   - {tool_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: ベクトル検索ツールの動作確認
# MAGIC
# MAGIC ツールが正しく動作するか、実際に検索してみます。

# COMMAND ----------

# テスト検索を実行
test_query = "このマニュアルはどういった目的で作成されていますか？"
search_results = vector_search_tool.invoke(input=test_query)

print(f"検索クエリ: {test_query}")
print(f"\n検索結果の一部:")
print(search_results[:500] + "..." if len(search_results) > 500 else search_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: エージェントの作成
# MAGIC
# MAGIC ### エージェントとは
# MAGIC
# MAGIC エージェントは、LLMとツールを組み合わせて、自律的に問題を解決するシステムです。
# MAGIC
# MAGIC ### 動作の流れ
# MAGIC
# MAGIC 1. **質問を受け取る**: ユーザーからの質問を理解
# MAGIC 2. **計画を立てる**: どのツールを使うべきか判断
# MAGIC 3. **ツールを実行**: 必要なツールを呼び出す
# MAGIC 4. **結果を解釈**: ツールの実行結果を理解
# MAGIC 5. **回答を生成**: 最終的な回答を作成
# MAGIC
# MAGIC ### システムプロンプトの役割
# MAGIC
# MAGIC システムプロンプトは、エージェントの「性格」と「行動指針」を定義します。
# MAGIC これにより、エージェントの振る舞いを制御できます。
# MAGIC
# MAGIC **今回のシステムプロンプトの特徴**:
# MAGIC - 医療アシスタントとしての役割を明確化
# MAGIC - ツールの使用タイミングを指示
# MAGIC - エラーハンドリングの方法を定義
# MAGIC - 医療外の質問への対応方法を指示

# COMMAND ----------

from langchain.agents import create_agent
from IPython.display import Markdown, display

# ========================================
# システムプロンプト（エージェントの性格と行動指針）
# ========================================

SYSTEM_PROMPT = """あなたは優秀な医療アシスタントです。
患者さんやお医者さんからのあらゆる医療に関する質問に対して、的確かつプロフェッショナルに日本語で回答します。

【医療に関する質問への対応】
- まずあなたの持っている知識のみで回答できるか、それともツールを使うかを判断してください。ただし、「がん（Cancer）の治療」に関しては、必ず最新の医療マニュアルを参照ください。
- ツールを使う場合はその実行結果を得た上で、次のアクションを決めてください

【医療マニュアル検索する場合】
- がんに関する問い合わせについては、必ず最新の医療マニュアルを検索し、検索結果に基づいてのみ回答してください
- 検索結果に答えが見つからない場合：
  1. 検索クエリを変えて数回試してください
  2. それでも見つからなければ、正直に「わからない」と伝えてください

【患者データを取得する場合】
- get_patient_data関数を使って、患者IDから特徴量データを取得します
- 取得したデータは30次元の配列形式です

【乳がん予測を行う場合】
- predict_cancer関数を使って、患者の乳がんリスクを予測します
- この関数は患者IDと30次元の特徴量を引数として受け取ります
- 予測結果には悪性(malignant)と良性(benign)の確率が含まれます

【挨拶への対応】
- 丁寧に挨拶を返してください
- 簡潔に自己紹介してください
- 医療に関する質問を促してください

【その他の質問への対応】
- 回答できない旨をはっきりと伝えてください
- 簡潔に自己紹介してください
- 医療に関する質問を促してください

【回答の心得】
- 質問者の意図をしっかりと理解してください
- わかりやすく丁寧に説明してください
- 専門用語は必要に応じて説明を加えてください"""

# ========================================
# エージェントの作成
# ========================================

# すべてのツールをリストにまとめる
tools = [vector_search_tool]  # ベクトル検索ツール
tools.extend(uc_toolkit.tools)  # Unity Catalog関数ツール

# エージェントを作成
agent = create_agent(
    model,              # 使用するLLM
    tools,              # 利用可能なツールのリスト
    system_prompt=SYSTEM_PROMPT  # エージェントの行動指針
)

print("✓ エージェントを作成しました")
print(f"   使用可能なツール数: {len(tools)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9: エージェントワークフローの可視化
# MAGIC
# MAGIC 構築したエージェントのワークフローを図で確認します。
# MAGIC
# MAGIC ### 図の見方
# MAGIC
# MAGIC - **四角形**: ノード（処理）
# MAGIC - **矢印**: エッジ（処理の流れ）
# MAGIC - **菱形**: 条件分岐
# MAGIC
# MAGIC この図を見ることで、エージェントがどのように動作するか視覚的に理解できます。

# COMMAND ----------

from IPython.display import Image, display

try:
    # エージェントのグラフ構造を可視化
    graph_image = agent.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    print("✓ ワークフローの図を表示しました")
except Exception as e:
    print(f"図の表示に失敗しました: {e}")
    print("（この機能は環境によっては動作しない場合があります）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ10: エージェントのテスト
# MAGIC
# MAGIC いよいよ構築したエージェントを動かしてみます！
# MAGIC
# MAGIC ### テストケースの概要
# MAGIC
# MAGIC 以下の4つのシナリオでエージェントをテストします：
# MAGIC
# MAGIC 1. **患者データと予測**: 患者IDから乳がんリスクを予測
# MAGIC 2. **挨拶**: 基本的な対話能力を確認
# MAGIC 3. **範囲外の質問**: 医療外の質問への適切な対応を確認
# MAGIC 4. **複合的な質問**: 複数のツールを組み合わせた処理を確認

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース1: ベクトル検索を伴う回答
# MAGIC
# MAGIC **目的**: エージェントが適切に質問に、ベクトル検索を実施した上で対応できるか確認

# COMMAND ----------

# 挨拶
res = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "食道がんの主な治療法は何ですか？"
    }]
})

# 結果をMarkdown形式で表示
display(Markdown(res["messages"][-1].content))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース2: 患者データと乳がん予測
# MAGIC
# MAGIC **目的**: エージェントが以下を実行できるか確認
# MAGIC 1. 患者IDを理解
# MAGIC 2. get_patient_data関数で特徴量を取得
# MAGIC 3. predict_cancer関数で予測を実行
# MAGIC 4. 結果を分かりやすく説明
# MAGIC
# MAGIC **期待される動作**:
# MAGIC - 患者P000123のデータを取得
# MAGIC - 乳がんの悪性/良性の確率を予測
# MAGIC - 結果を医療的な観点から説明

# COMMAND ----------

# 患者の乳がんリスク予測を依頼
res = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "P000190の患者は乳がんの悪性の可能性がありますか？"
    }]
})

# 結果をMarkdown形式で表示
display(Markdown(res["messages"][-1].content))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース3: 医療外の質問
# MAGIC
# MAGIC **目的**: エージェントが適切に範囲外の質問を断れるか確認
# MAGIC
# MAGIC **期待される動作**:
# MAGIC 1. 回答できない旨を丁寧に伝える
# MAGIC 2. 自己紹介をする（医療アシスタントであることを再確認）
# MAGIC 3. 医療に関する質問を促す

# COMMAND ----------

# 医療外の質問
res = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "親友と喧嘩しました。どうすれば仲直りできる？"
    }]
})

# 結果をMarkdown形式で表示
display(Markdown(res["messages"][-1].content))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース4: 複合的な医療質問
# MAGIC
# MAGIC **目的**: エージェントが複数のツールを組み合わせて複雑な質問に答えられるか確認
# MAGIC
# MAGIC **期待される動作**:
# MAGIC 1. 患者データを取得（get_patient_data）
# MAGIC 2. 乳がん予測を実行（predict_cancer）
# MAGIC 3. 医療マニュアルを検索（search_medical_manual）
# MAGIC 4. すべての情報を統合して包括的なアドバイスを提供

# COMMAND ----------

# 複雑な医療質問（複数のツールを組み合わせた処理）
res = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "P000190の患者は現状はどうでしょうか？その状況に応じて最新の医療マニュアルに基づきアドバイスをお願いします"
    }]
})

# 結果をMarkdown形式で表示
display(Markdown(res["messages"][-1].content))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # フェーズ2: エージェントの評価
# MAGIC ---
# MAGIC
# MAGIC ## なぜ評価が重要なのか
# MAGIC
# MAGIC エージェントを構築しただけでは、本番環境で使用できません。
# MAGIC 以下の理由から、客観的な評価が不可欠です：
# MAGIC
# MAGIC ### 1. 品質保証
# MAGIC - 手動テストだけでは見逃すケースがある
# MAGIC - 様々なシナリオで一貫した品質を確保
# MAGIC - 本番投入前に問題を発見
# MAGIC
# MAGIC ### 2. 継続的改善
# MAGIC - システムプロンプトの変更による影響を測定
# MAGIC - 新しいツール追加の効果を確認
# MAGIC - リグレッション（性能劣化）を防止
# MAGIC
# MAGIC ### 3. ステークホルダーへの説明
# MAGIC - 「このエージェントは信頼できる」という根拠を提示
# MAGIC - ビジネス価値を数値で示す
# MAGIC - 投資対効果を証明
# MAGIC
# MAGIC ### 4. 本番環境でのモニタリング
# MAGIC - 実際の使用状況での性能を追跡
# MAGIC - 問題が発生したときに早期に検知
# MAGIC - データドリフトに対応
# MAGIC
# MAGIC これから、MLflowを使ってエージェントを自動的に評価する方法を学びます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ11: 評価用のラッパー関数を定義
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
    res = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    
    # 最後のメッセージ（エージェントの回答）を返す
    return res["messages"][-1].content

print("✅ 評価用ラッパー関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ12: 評価データセットの作成
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
# MAGIC ## ステップ13: 評価指標（スコアラー）の定義
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
# MAGIC ## ステップ14: エージェントの評価を実行
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
mlflow.set_experiment(f"/Workspace/Users/{username}/medical_agent_evaluation")
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
# MAGIC ## ステップ15: 評価結果の確認
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
# MAGIC ## まとめ：ハンズオンの振り返り
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC このハンズオンで、以下のスキルを習得しました：
# MAGIC
# MAGIC #### 1. LLMの基本的な使い方
# MAGIC - Databricks上のLLMエンドポイントへの接続
# MAGIC - ChatDatabricksクラスの使用方法
# MAGIC - システムプロンプトによる動作制御
# MAGIC
# MAGIC #### 2. 複数のツールの統合
# MAGIC - ベクトル検索ツールの設定と使用
# MAGIC - Unity Catalog関数のツール化
# MAGIC - 異なる種類のツールを1つのエージェントで使用
# MAGIC
# MAGIC #### 3. エージェントの構築
# MAGIC - LangChainによるエージェント作成
# MAGIC - システムプロンプトによる行動制御
# MAGIC - ツールの選択と実行の自動化
# MAGIC
# MAGIC #### 4. 実用的なエージェントの実装
# MAGIC - 複数のツールを使いこなすエージェント
# MAGIC - 適切なエラーハンドリング
# MAGIC - ユーザーフレンドリーな応答
# MAGIC
# MAGIC #### 5. エージェントの評価とモニタリング ← 新規追加
# MAGIC - MLflowを使った自動評価の実装
# MAGIC - 評価データセットの設計方法
# MAGIC - LLMジャッジによる品質評価
# MAGIC - Correctness（正確性）、RetrievalSufficiency（検索十分性）などの指標
# MAGIC - 評価結果の分析と改善への活用
# MAGIC
# MAGIC ### エージェント開発のライフサイクル
# MAGIC
# MAGIC 今回のハンズオンで、エージェント開発の完全なサイクルを体験しました：
# MAGIC
# MAGIC 1. **設計**: どんなツールが必要かを決定
# MAGIC 2. **実装**: LangChainでエージェントを構築
# MAGIC 3. **テスト**: 手動で動作を確認
# MAGIC 4. **評価**: 自動評価で品質を測定
# MAGIC 5. **改善**: 評価結果を基に最適化
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC このエージェントをさらに改善するには：
# MAGIC
# MAGIC 1. **より多くのツールを追加**: 薬剤情報、検査結果の解釈など
# MAGIC 2. **システムプロンプトの最適化**: より正確で適切な応答を実現
# MAGIC 3. **評価データセットの拡充**: より多様なシナリオをカバー
# MAGIC 4. **評価指標の追加**: Safety（安全性）、RelevanceToQuery（関連性）など
# MAGIC 5. **エラーハンドリングの強化**: より堅牢なエージェントに
# MAGIC 6. **本番環境へのデプロイ**: Model Servingエンドポイントとして公開
# MAGIC 7. **継続的な評価**: 本番環境でのパフォーマンスをモニタリング

# COMMAND ----------

# MAGIC %md
# MAGIC ## おわりに
# MAGIC
# MAGIC お疲れ様でした！このハンズオンを通じて、実用的なAIエージェントの
# MAGIC 構築から評価までの一連のプロセスを学びました。
# MAGIC
# MAGIC ### 作成したもの
# MAGIC
# MAGIC #### ツール層
# MAGIC - ✅ 医療マニュアルを検索できるベクトル検索ツール
# MAGIC - ✅ 患者データを取得できるUnity Catalog関数
# MAGIC - ✅ 乳がんリスクを予測できるML関数
# MAGIC
# MAGIC #### エージェント層
# MAGIC - ✅ これらすべてを統合した医療アシスタントエージェント
# MAGIC - ✅ 適切なツール選択と実行ができる自律的なシステム
# MAGIC
# MAGIC #### 評価層（新規追加）
# MAGIC - ✅ 評価データセットとテストケース
# MAGIC - ✅ MLflowを使った自動評価パイプライン
# MAGIC - ✅ LLMジャッジによる品質スコアリング
# MAGIC - ✅ トレーシングとモニタリングの仕組み
# MAGIC
# MAGIC ### 本番環境への適用
# MAGIC
# MAGIC このハンズオンで学んだ評価の仕組みは、本番環境でも重要です：
# MAGIC
# MAGIC - **デプロイ前**: 評価スコアが基準を満たしているか確認
# MAGIC - **A/Bテスト**: 新旧バージョンのエージェントを比較
# MAGIC - **継続的モニタリング**: 本番環境での回答品質を監視
# MAGIC - **リグレッション防止**: 改善が本当に改善になっているか検証
# MAGIC
# MAGIC ### 活用できる場面
# MAGIC
# MAGIC このエージェントの設計パターンは、医療以外の様々な分野でも応用できます：
# MAGIC - カスタマーサポート（FAQ検索 + データベース検索）
# MAGIC - 営業支援（顧客データ取得 + 提案生成）
# MAGIC - 人事業務（人事規定検索 + 従業員情報取得）
# MAGIC - 金融アドバイス（市場データ検索 + リスク予測）
# MAGIC
# MAGIC 重要なのは、どの分野でも**評価**が不可欠だということです。
# MAGIC エージェントを本番環境に投入する前に、必ず品質を測定し、
# MAGIC 継続的に改善していく仕組みを作りましょう。
# MAGIC
# MAGIC ぜひ、あなたの業務に合わせてカスタマイズしてみてください！
