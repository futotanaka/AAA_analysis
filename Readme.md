# AAA_segmentation

腹部大動脈瘤（Abdominal Aortic Aneurysm, AAA）のCT画像に対する**自動セグメンテーションおよび解析**を行うためのPythonコードをまとめたものです．  
医用画像処理の基礎モデルである **U-Net系ネットワーク** を中心に，学習，予測，前処理，評価，および血管やステントの形態解析を行う機能を提供します．  

## 開発・動作環境
- CPU: Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz (2.90 GHz)
- メモリ: 64.0 GB
- OS: WSL Ubuntu-20.04
- コンパイラ: Python 3.10.12
- ライブラリ:  
    - matplotlib 3.8.0
    - numpy 1.26.0
    - opencv-python 4.8.1.78
    - scipy 1.11.3
    - torch 2.1.2
- ツール: Visual Studio Code 1.102.3

## プロジェクト全体の流れ

1. **データ準備**
   - 画像（.mhd形式）を読み込み
   - 学習用・予測用データセットを作成、データ増強を行う
   -  前処理用関数
   - .mhdファイルを操作するための関数 

2. **モデル**
   - U-Net，U-Net++，Attention U-Net などのセグメンテーションモデルを定義

3. **学習**
   - 画像とラベルを使ってモデルを訓練
   - Dice係数などの損失関数を利用
   - テストデータセットでモデル性能を検証

4. **予測**
   - 学習済みモデルを用いて未知の画像をセグメンテーション、結果を保存・可視化

5. **解析**
   - arterial_analysis.py -> 血管の分岐スライスと動脈瘤範囲を決定、動脈瘤最大短径と体積を計算
   - branch_segmentation.py -> ステントグラフトの各部分の分割
   - stent_analysis.py -> ステントグラフトのboundingboxと曲率特徴量、中心線長さを計測

---

## ファイル構成と役割

### モデル関連
- **`unet.py`**  
  基本的なU-Netモデルの定義
- **`Unet_plus.py`**  
  U-Net++の定義
- **`attUnet.py`**  
  Attention U-Netの定義  

### 学習
- **`training.py`**  
  モデルを学習させるメインスクリプト
  - データセットの読み込み  
  - モデルの選択  
  - 学習の実行  
- **`test.py`**  
  テストデータセットでモデル性能を検証
- **`prediction.py`**  
  学習済みモデルを使って新しいCT画像をセグメンテーション

### データ処理
- **`dataExtractor.py`**  
  生データから必要な部分を切り出す  
- **`dataSetCreat.py`**  
  学習用データセットを作成する  
- **`predDataExtractor.py`**  
  推論用データを準備するためのスクリプト  
- **`predDataSetCreat.py`**  
  推論用データセットを作成する  
- **`preprocessing.py`**  
  前処理（正規化やサイズ変換など）を行う  
- **`mhd_io.py`**  
  `.mhd`ファイルの読み書きをサポートする  

### 評価・損失関数
- **`dice.py`**  
  Dice lossの損失関数
- **`diceCE.py`**  
  Dice + Cross Entropy の複合損失関数

### 解析
- **`skeleton_test.py`**  
  .mhdファイルから解析する実行ファイル  
- **`arterial_analysis.py`**  
  血管の分岐スライスと動脈瘤範囲を決定、動脈瘤最大短径と体積を計算  
- **`branch_segmentation.py`**  
  ステントグラフトの各部分の分割
- **`stent_analysis.py`**  
  ステントグラフトのboundingboxと曲率特徴量、中心線長さを計測

### その他
- **`overlay.py`**  
  画像とセグメンテーション結果を重ねて表示  

---

## 使用方法

1. **環境整備**
2. **データ準備**  
以下のようにデータを配置する(basepath: train,validationなど)  
    ```
    basepath/  
    └─ 症例ID0/   
        └─ 日付番号/   
            ├─ original.mhd  # 元CT画像
            ├─ original.raw  
            ├─ stent_mask.mhd  # ステントグラフト領域正解ラベル
            ├─ stent_mask.raw  
            ├─ vol000-label.mhd  # 血管領域正解ラベル
            └─ vol000-label.raw  
    └─ 症例ID1/  
    └─ 日付番号/  
3. **生データ処理**
    ```
    python dataExtractor.py <origin_dataset> <output_data>
    ```
    - **`origin_dataset`**: 元の画像データが保存されているディレクトリ  
    - **`output_data`**: 処理後データの保存先  
    保存された処理後データはmasksとoriginalこの二つのフォルダに保存される、処理後データは自動消去されないため、異なるデータを同じところに保存する時に`rm -rf`を利用して古いデータを消去する必要がある。
4. **学習**
    ```
    python training.py <training_dataset> <validation_dataset> <output_path> -mo <モデル選択> -m <最大epoch数> -a
    ```
    - **`training_dataset`**: trainingデータの処理後データの保存先
    - **`validation_dataset`**: validationデータの処理後データの保存先
    - **`output_path`**: 学習中に出力されるログやモデルの保存先  
    
    オプション引数：
    - **`-mo`** : モデル選択（例: 2 = U-Net++）  
    - **`-m`** : 学習エポック数（例: 100）  
    - **`-a`**     : データ増強を有効化  

    出力先は`output_path`配下の`csvFiles`（ログ等）と`models`（学習済みモデル）です。  
    パラメータ探索範囲：
    #### 固定パラメータ

    | パラメータ              | 範囲        |
    |-------------------------|-------------|
    | Loss関数                | Dice loss   |
    | Optimizer               | Adam        |
    | Epoch数                 | 100         |
    | Early stopping patience | 10          |
    ---
    #### パラメータ探索範囲

    | パラメータ             | 範囲            |
    |------------------------|-----------------|
    | Learning rate          | 10^n, n=[-6,-2] |
    | β₁(Adam)              | [0.9, 0.99]     |
    | U-Net++ filter数       | 2^n, n=[2,8]    |
    | Batch size             | 2^n, n=[1,3]    |

    ```
    python test.py <input_dir> <model_path> <output_dir> <slice_dir> -f <filter数> -mo <model選択> -t <threshold>
    ```
    - **`input_dir`**: データの保存先（上記 dataExtractor.py の出力）
    - **`model_path`**: 学習済みモデル（.pth ファイル）のパス
    - **`output_dir`**: 抽出結果の保存先（ログ、.mhdファイル）
    - **`-f`** : filter数
    - **`-mo`** : モデル選択
    
    オプション引数：
    - **`-t`**     : 予測マスクのしきい値
5. **予測**
    ```
    python predDataExtractor.py <input_dir> <output_dir>
    ```
    - **`input_dir`**: 元の画像データが保存されているディレクトリ
    - **`output_dir`**: 処理後データの保存先
    ```
    python prediction.py <input_dir> <model_path> <output_dir> <slice_dir> -f <filter数> -mo <model選択> -t <threshold>
    ```
    - **`input_dir`**: データの保存先（上記 predDataExtractor.py の出力）
    - **`model_path`**: 学習済みモデル（.pth ファイル）のパス
    - **`output_dir`**: 抽出結果の保存先（.mhdファイル）
    - **`-f`** : filter数
    - **`-mo`** : モデル選択
    
    オプション引数：
    - **`-t`**     : 予測マスクのしきい値
6. **解析**
    ```bash
    python skeleton_test.py <input_dir> <output_dir>
    ```
    - **`input_dir`**: 入力ファイルの保存先
    - **`output_dir`**: 出力ファイルの保存先  

    **入力ファイル**：中身は [データ準備](#使用方法) と同じフォーマットでデータを配置するか、`<症例ID>_<日付番号>_prediction.mhd` という形式で命名されたファイルを配置してください。なお、正解ラベルと予測結果の `.mhd` ファイルを同時に使用することには対応していません。
    
    **出力ファイル**：入力ファイル1つに対して結果ファイルが4つ生成されます。
    - **`<症例ID>_<日付番号>_arterial.mhd`**: 血管領域。楕円近似の結果と最大短径が計算されたスライスに表示されます。  
    - **`<症例ID>_<日付番号>_segment_mask.mhd`**: 分割されたステントグラフト各部位（ラベル番号：1-3）と血管領域（ラベル番号：10）が含まれます。  
    - **`<症例ID>_<日付番号>_skeleton_arterial.mhd`**: 細線化後の血管領域。  
    - **`<症例ID>_<日付番号>_skeleton_stent.mhd`**: 細線化後のステントグラフト各部位（ラベル番号:1-3）と、中心線長さに加算された部分（ラベル番号:20）が含まれます。重複部分があるため、ラベル番号20を除外したい場合は、以下の2行をコメントアウトしてください（`stent_analysis.py` 内）。
        ```
        for z_local, x, y in visited:
            visited_mask[z0 + z_local, x, y] = 20
    各項目の計測値はコマンドライン出力に加えて、`skeleton_test.py` が存在するフォルダ内の `output.txt` に保存されます。
