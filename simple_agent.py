"""
医療アシスタントエージェント (agent.py)

このファイルは、04.agent_evalノートブックから呼び出され、
MLflowモデルとしてデプロイされるエージェントを定義します。
"""

import os
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    UCFunctionToolkit,
)
from langchain.agents import create_agent
from IPython.display import Markdown, display

###############################################################################
# セクション1: 環境変数から設定を読み込む
###############################################################################

# 環境変数を取得（04.agent_evalで設定される）
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME")
VS_NAME = os.environ.get("VS_NAME")
UC_TOOL_NAMES = os.environ.get("UC_TOOL_NAMES", "")

# 設定確認用の出力
print("=" * 50)
print("エージェントの設定:")
print("=" * 50)
print(f"LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
print(f"VS_NAME: {VS_NAME}")
print(f"UC_TOOL_NAMES: {UC_TOOL_NAMES}")
print("=" * 50)

# MLflowの自動ロギングを有効化
mlflow.langchain.autolog()

###############################################################################
# セクション2: LLMの初期化
###############################################################################

# Databricks上のLLMエンドポイントに接続
model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

print(f"✓ LLMモデルを初期化しました: {LLM_ENDPOINT_NAME}")

###############################################################################
# セクション3: システムプロンプトの定義
###############################################################################

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

###############################################################################
# セクション4: ツールの作成
###############################################################################

tools = []

# ========================================
# ツール1: ベクトル検索ツール
# ========================================

if VS_NAME:
    try:
        vector_search_tool = VectorSearchRetrieverTool(
            index_name=VS_NAME,
            tool_name="search_medical_manual",
            num_results=10,  # 上位10件の関連文書を取得
            tool_description="最新の医療マニュアルを検索します。医療や治療方法に関する質問に答えるために使用してください。",
            disable_notice=True
        )
        tools.append(vector_search_tool)
        print(f"✓ ベクトル検索ツールを追加: {VS_NAME}")
    except Exception as e:
        print(f"⚠ Warning: ベクトル検索ツールの追加に失敗: {e}")

# ========================================
# ツール2・3: Unity Catalog関数ツール
# ========================================

if UC_TOOL_NAMES:
    try:
        # カンマ区切りまたはワイルドカードパターンをリストに変換
        if "," in UC_TOOL_NAMES:
            tool_names = [name.strip() for name in UC_TOOL_NAMES.split(",") if name.strip()]
        else:
            tool_names = [UC_TOOL_NAMES]
        
        # Unity Catalog関数をツールとして登録
        uc_toolkit = UCFunctionToolkit(function_names=tool_names)
        tools.extend(uc_toolkit.tools)
        
        print(f"✓ Unity Catalog関数ツールを追加:")
        for tool_name in tool_names:
            print(f"  - {tool_name}")
    except Exception as e:
        print(f"⚠ Warning: Unity Catalog関数ツールの追加に失敗: {e}")

print(f"\n合計 {len(tools)} 個のツールを作成しました\n")

###############################################################################
# セクション5: エージェントの作成
###############################################################################

# create_agent関数を使ってシンプルにエージェントを作成
# この関数は、LangGraphのワークフローを自動的に構築します
agent = create_agent(
    model,                      # 使用するLLM
    tools,                      # 利用可能なツールのリスト
    system_prompt=SYSTEM_PROMPT # エージェントの行動指針
)

print("✓ エージェントを作成しました")
print(f"  使用可能なツール数: {len(tools)}")
print("=" * 50 + "\n")

###############################################################################
# セクション7: MLflowにエージェントを登録
###############################################################################

# エージェントをラップ
AGENT = agent

# MLflowに推論時に使用するモデルとして登録
mlflow.models.set_model(AGENT)

print("✅ エージェントの準備が完了しました\n")