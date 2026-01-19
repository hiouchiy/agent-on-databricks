# Databricks notebook source
# MAGIC %md
# MAGIC # Unity Catalog AI ツールの準備
# MAGIC
# MAGIC ## このノートブックで行うこと
# MAGIC
# MAGIC このノートブックでは、AIエージェントが使用できる3つのツール（関数）をUnity Catalogに登録します。
# MAGIC 具体的には以下の3つのツールを作成します：
# MAGIC
# MAGIC 1. **get_todays_date（Python関数）**
# MAGIC    現在の日付を取得する関数
# MAGIC
# MAGIC 2. **get_patient_data（SQL関数）**
# MAGIC    患者IDから乳がん特徴量データを取得する関数
# MAGIC
# MAGIC 3. **predict_cancer（Python関数）**
# MAGIC    機械学習モデルを使って乳がんの予測を行う関数
# MAGIC
# MAGIC ## なぜUnity Catalogにツールを登録するのか
# MAGIC
# MAGIC Unity Catalogにツールを登録することで：
# MAGIC - AIエージェントが自動的にこれらの関数を呼び出せるようになります
# MAGIC - 関数のバージョン管理や権限管理が容易になります
# MAGIC - 複数のユーザーやプロジェクトで同じツールを再利用できます

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 必要なパッケージのインストール
# MAGIC
# MAGIC `requirements.txt` に記載されているパッケージをインストールします。
# MAGIC インストール後、Pythonを再起動して新しいパッケージを使えるようにします。

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # パッケージをPython環境にロードするために再起動します
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: 環境設定
# MAGIC
# MAGIC カタログとスキーマを設定します。
# MAGIC ユーザー名に基づいて一意のカタログ名を生成し、データの分離を実現します。

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

# 設定完了メッセージを表示
print(f"✅ 環境セットアップ完了")
print(f"   カタログ: {CATALOG}")
print(f"   スキーマ: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: ツール1 - 日付取得関数の定義
# MAGIC
# MAGIC ### get_todays_date 関数
# MAGIC
# MAGIC この関数は今日の日付を 'YYYY-MM-DD' 形式で返します。
# MAGIC AIエージェントが日付情報を必要とする場合に使用されます。
# MAGIC
# MAGIC **使用例**:
# MAGIC - 「今日の日付で記録を保存してください」
# MAGIC - 「今日受診した患者のデータを取得してください」

# COMMAND ----------

def get_todays_date() -> str:
    """
    今日の日付を 'YYYY-MM-DD' 形式で返します。
    
    Returns:
        str: 今日の日付を 'YYYY-MM-DD' 形式で返します。
    """
    from datetime import datetime
    # 現在の日時を取得し、YYYY-MM-DD形式の文字列に変換
    return datetime.now().strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: 関数のテスト
# MAGIC
# MAGIC 定義した関数が正しく動作するか確認します。

# COMMAND ----------

# 関数を実行してテスト
today = get_todays_date()
today  # 結果を表示

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: Unity Catalogへのデプロイ（Python関数）
# MAGIC
# MAGIC ### DatabricksFunctionClient とは
# MAGIC
# MAGIC Unity CatalogにPython関数を登録するためのクライアントです。
# MAGIC 関数のメタデータ（名前、引数、戻り値の型、説明）を自動的に抽出し、
# MAGIC Unity Catalogに登録します。
# MAGIC
# MAGIC ### デプロイのメリット
# MAGIC
# MAGIC - AIエージェントが関数を自動検出できる
# MAGIC - 関数の説明（docstring）が自動的にメタデータとして登録される
# MAGIC - バージョン管理と権限管理が可能

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

# Unity Catalogクライアントを初期化
client = DatabricksFunctionClient()

# get_todays_date関数をUnity Catalogにデプロイ
# - func: デプロイする関数
# - catalog: 登録先のカタログ名
# - schema: 登録先のスキーマ名
# - replace=True: 既に同名の関数がある場合は上書き
python_tool_uc_info = client.create_python_function(
    func=get_todays_date, 
    catalog=CATALOG, 
    schema=SCHEMA, 
    replace=True
)

# ツールはUC内の `{catalog}.{schema}.{func}` という名前の関数にデプロイされます
# ここで {func} は関数の名前です

# デプロイされたUnity Catalog関数名を表示します
display(f"デプロイされたUnity Catalog関数名: {python_tool_uc_info.full_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: ツール2 - 患者データ取得関数の作成（SQL関数）
# MAGIC
# MAGIC ### get_patient_data 関数
# MAGIC
# MAGIC この関数は、患者IDを受け取り、その患者の乳がん特徴量（30次元）を
# MAGIC Deltaテーブルから取得して配列形式で返します。
# MAGIC
# MAGIC ### 関数の仕様
# MAGIC
# MAGIC **引数**:
# MAGIC - `p_patient_id`: 患者ID（文字列、例: "P000123"）
# MAGIC
# MAGIC **戻り値**:
# MAGIC - 30次元の特徴量配列（ARRAY型）
# MAGIC - 特徴量の順序は機械学習モデルの入力順序と一致
# MAGIC
# MAGIC ### 30次元の特徴量とは
# MAGIC
# MAGIC scikit-learnの乳がんデータセットに基づく特徴量：
# MAGIC - **mean_〜**: 腫瘍の平均的な特徴（10個）
# MAGIC - **〜_error**: 測定誤差の標準偏差（10個）
# MAGIC - **worst_〜**: 最悪の（最大の）値（10個）
# MAGIC
# MAGIC 各カテゴリには以下の測定値が含まれます：
# MAGIC - radius（半径）、texture（テクスチャ）、perimeter（周囲）、area（面積）
# MAGIC - smoothness（滑らかさ）、compactness（コンパクト性）、concavity（凹み）
# MAGIC - concave points（凹点）、symmetry（対称性）、fractal_dimension（フラクタル次元）

# COMMAND ----------

# SQL関数を作成
sql = f"""CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_patient_data(
    p_patient_id STRING COMMENT '患者ID。例: P000123'
)
RETURNS ARRAY<DOUBLE>
COMMENT '患者IDに紐づく乳がん30特徴量をDeltaから取得し、モデル入力順のARRAY(30次元)で返す。'
LANGUAGE SQL
RETURN (
    SELECT array(
        -- ========================================
        -- 平均値系の特徴量（10個）
        -- ========================================
        cast(mean_radius as double),              -- 1. 腫瘍の平均半径
        cast(mean_texture as double),             -- 2. 平均テクスチャ（グレースケールの標準偏差）
        cast(mean_perimeter as double),           -- 3. 平均周囲長
        cast(mean_area as double),                -- 4. 平均面積
        cast(mean_smoothness as double),          -- 5. 平均滑らかさ（半径の局所変動）
        cast(mean_compactness as double),         -- 6. 平均コンパクト性（周囲^2 / 面積 - 1.0）
        cast(mean_concavity as double),           -- 7. 平均凹み度（輪郭の凹んでいる部分の深刻度）
        cast(mean_concave_points as double),      -- 8. 平均凹点数（輪郭の凹んでいる部分の数）
        cast(mean_symmetry as double),            -- 9. 平均対称性
        cast(mean_fractal_dimension as double),   -- 10. 平均フラクタル次元（"coastline approximation" - 1）
        
        -- ========================================
        -- 誤差系の特徴量（標準偏差、10個）
        -- ========================================
        cast(radius_error as double),             -- 11. 半径の標準誤差
        cast(texture_error as double),            -- 12. テクスチャの標準誤差
        cast(perimeter_error as double),          -- 13. 周囲長の標準誤差
        cast(area_error as double),               -- 14. 面積の標準誤差
        cast(smoothness_error as double),         -- 15. 滑らかさの標準誤差
        cast(compactness_error as double),        -- 16. コンパクト性の標準誤差
        cast(concavity_error as double),          -- 17. 凹み度の標準誤差
        cast(concave_points_error as double),     -- 18. 凹点数の標準誤差
        cast(symmetry_error as double),           -- 19. 対称性の標準誤差
        cast(fractal_dimension_error as double),  -- 20. フラクタル次元の標準誤差
        
        -- ========================================
        -- 最悪値系の特徴量（10個）
        -- ========================================
        cast(worst_radius as double),             -- 21. 最大半径
        cast(worst_texture as double),            -- 22. 最大テクスチャ
        cast(worst_perimeter as double),          -- 23. 最大周囲長
        cast(worst_area as double),               -- 24. 最大面積
        cast(worst_smoothness as double),         -- 25. 最大滑らかさ
        cast(worst_compactness as double),        -- 26. 最大コンパクト性
        cast(worst_concavity as double),          -- 27. 最大凹み度
        cast(worst_concave_points as double),     -- 28. 最大凹点数
        cast(worst_symmetry as double),           -- 29. 最大対称性
        cast(worst_fractal_dimension as double)   -- 30. 最大フラクタル次元
    )
    FROM {CATALOG}.{SCHEMA}.patient_breast_cancer_features
    WHERE patient_id = p_patient_id  -- 指定された患者IDのデータを取得
    LIMIT 1  -- 1件のみ取得（patient_idはユニークと想定）
);"""

# SQL関数を実行して登録
spark.sql(sql)

print(f"✅ SQL関数を作成しました: {CATALOG}.{SCHEMA}.get_patient_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: ツール3 - 乳がん予測関数の定義
# MAGIC
# MAGIC ### predict_cancer 関数
# MAGIC
# MAGIC この関数は、30次元の特徴量ベクトルを機械学習モデルに送信し、
# MAGIC 乳がんの予測結果（良性/悪性の確率）を返します。
# MAGIC
# MAGIC ### 関数の動作フロー
# MAGIC
# MAGIC 1. **入力検証**: 特徴量が30次元であることを確認
# MAGIC 2. **データ整形**: モデルが期待する形式（列名付きデータフレーム）に変換
# MAGIC 3. **モデル呼び出し**: Databricks Model Servingエンドポイントにリクエスト送信
# MAGIC 4. **結果整形**: 予測結果をJSON形式で返却
# MAGIC
# MAGIC ### エラーハンドリング
# MAGIC
# MAGIC - 特徴量がNoneの場合: "not_found"エラーを返す
# MAGIC - 特徴量の次元数が30でない場合: "invalid_feature_length"エラーを返す
# MAGIC - モデル呼び出しが失敗した場合: 例外を発生させる
# MAGIC
# MAGIC ### セキュリティに関する注意
# MAGIC
# MAGIC **重要**: 本番環境では、認証情報（HOST/TOKEN）を以下の方法で管理してください：
# MAGIC - Databricks Secretsを使用
# MAGIC - 環境変数を使用
# MAGIC - コードに直接記述しない

# COMMAND ----------

def predict_cancer(patient_id: str, features: list[float]) -> str:
    """
    【目的】
    乳がん判定モデル（Databricks Model Serving）に対して、30次元の特徴量ベクトルを送信し、
    患者IDと推論結果（malignant / benign の確率）をJSON文字列で返します。
    
    【前提】
    - features は「モデル学習時と同じ順序」で並んだ 30 個の特徴量であること
    - Databricks Model Serving のエンドポイントが作成済みであること
    - 認証情報（DATABRICKS_HOST / DATABRICKS_TOKEN）が安全な方法で設定されていること
    
    ※ トークンをコードに直書きするのは避け、Secret や環境変数を利用してください
    
    Args:
        patient_id: 患者ID（例: "P000123"）
        features: 30次元の特徴量（順序固定）
    
    Returns:
        以下の形式の JSON 文字列を返します。
        - 正常系: 
          {"patient_id":"P000123","prediction":{"malignant":0.10,"benign":0.89}}
        - 異常系（データなし等）:
          {"patient_id":"P000123","error":"not_found"}
          {"patient_id":"P000123","error":"invalid_feature_length:29"}
    """
    import os, json, requests
    
    # ========================================
    # 入力検証
    # ========================================
    
    # 特徴量がNoneの場合はエラーを返す
    if features is None:
        return json.dumps(
            {"patient_id": patient_id, "error": "not_found"}, 
            ensure_ascii=False
        )
    
    # 特徴量が30次元でない場合はエラーを返す
    if len(features) != 30:
        return json.dumps(
            {"patient_id": patient_id, "error": f"invalid_feature_length:{len(features)}"}, 
            ensure_ascii=False
        )
    
    # ========================================
    # モデル入力データの準備
    # ========================================
    
    # モデルが期待する列名（スペース入り、30個）
    # scikit-learnの乳がんデータセットと同じ列名を使用
    columns = [
        # 平均値系（10個）
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
        
        # 誤差系（10個）
        "radius error", "texture error", "perimeter error", "area error", "smoothness error",
        "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
        
        # 最悪値系（10個）
        "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
        "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension",
    ]
    
    # モデルサービングAPIが期待する形式でペイロードを作成
    # dataframe_split形式: 列名とデータを分けて送信
    payload = {
        "dataframe_split": {
            "columns": columns,  # 列名のリスト
            "data": [features]   # データ（リストのリスト形式）
        }
    }
    
    # ========================================
    # 認証情報の取得
    # ========================================
    
    # 【重要】本番環境では以下のように環境変数やSecretから取得してください：
    # host = os.environ.get("DATABRICKS_HOST")
    # token = dbutils.secrets.get(scope="your-scope", key="databricks-token")
    
    # デモ用（実際の値に置き換えてください）
    host = "YOUR_HOST"    # 例: "https://your-workspace.cloud.databricks.com"
    token = "YOUR_TOKEN"  # 例: "dapi1234567890abcdef..."
    
    # 認証情報が設定されていない場合はエラー
    if not host or not token:
        raise RuntimeError("Missing DATABRICKS_HOST / DATABRICKS_TOKEN env vars")
    
    # ========================================
    # モデルサービングエンドポイントへのリクエスト
    # ========================================
    
    # エンドポイント名（実際のエンドポイント名に置き換えてください）
    serving_endpoint = "my-model-endpoint"
    
    # API URLを構築
    url = f"{host.rstrip('/')}/serving-endpoints/{serving_endpoint}/invocations"
    
    # HTTPリクエストを送信
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",      # 認証ヘッダー
            "Content-Type": "application/json"       # JSONリクエスト
        },
        data=json.dumps(payload),  # ペイロードをJSON文字列に変換
        timeout=30,                # タイムアウト30秒
    )
    
    # ========================================
    # レスポンスの処理
    # ========================================
    
    # HTTPステータスコードが300以上の場合はエラー
    if resp.status_code >= 300:
        raise RuntimeError(f"Serving call failed: {resp.status_code} {resp.text}")
    
    # レスポンスをJSONとしてパース
    out = resp.json()
    
    # 予測結果を取得（predictionsキーの最初の要素）
    # 形式: [malignant確率, benign確率]
    pred = out.get("predictions", [None])[0]
    
    # 結果をJSON文字列として返す
    return json.dumps({
        "patient_id": patient_id, 
        "prediction": {
            'malignant': pred[0],  # 悪性の確率
            'benign': pred[1]      # 良性の確率
        }
    }, ensure_ascii=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: predict_cancer関数のデプロイ
# MAGIC
# MAGIC 定義したpredict_cancer関数をUnity Catalogに登録します。
# MAGIC これにより、AIエージェントがこの関数を呼び出せるようになります。

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

# Unity Catalogクライアントを初期化
client = DatabricksFunctionClient()

# predict_cancer関数をUnity Catalogにデプロイ
py_info = client.create_python_function(
    func=predict_cancer,  # デプロイする関数
    catalog=CATALOG,      # 登録先のカタログ
    schema=SCHEMA,        # 登録先のスキーマ
    replace=True,         # 既存の関数を上書き
)

# デプロイ完了メッセージを表示
print(f"✅ Python関数を作成しました: {py_info.full_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC このノートブックでは、以下の3つのツールをUnity Catalogに登録しました：
# MAGIC
# MAGIC 1. ✅ **get_todays_date**: 現在の日付を取得
# MAGIC 2. ✅ **get_patient_data**: 患者IDから特徴量データを取得（SQL関数）
# MAGIC 3. ✅ **predict_cancer**: 機械学習モデルで乳がん予測を実行
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC これらのツールは、AIエージェントが以下のようなタスクを実行するために使用できます：
# MAGIC
# MAGIC - **日付取得**: 「今日の日付で記録を保存してください」
# MAGIC - **データ取得**: 「患者P000123のデータを取得してください」
# MAGIC - **予測実行**: 「この患者の乳がんリスクを予測してください」
# MAGIC
# MAGIC ### トラブルシューティング
# MAGIC
# MAGIC **エラーが出た場合**:
# MAGIC 1. カタログとスキーマが正しく作成されているか確認
# MAGIC 2. 必要なパッケージがインストールされているか確認
# MAGIC 3. Model Servingエンドポイントが作成されているか確認（predict_cancer関数用）
# MAGIC 4. 認証情報（HOST/TOKEN）が正しく設定されているか確認
# MAGIC
# MAGIC ### セキュリティのベストプラクティス
# MAGIC
# MAGIC - 本番環境では認証情報をDatabricks Secretsで管理
# MAGIC - Unity Catalogの権限設定で関数へのアクセスを制限
# MAGIC - モデルエンドポイントへのアクセス権限を適切に設定
