# DCT 紋理分類器第一版實驗報告

實驗日期：2026-09-13  
狀態：第一版完成（pseudo-label、訓練、獨立驗證、全圖推論、受控診斷、TextureSAM 相容性）

## 1. 實驗問題與範圍

本實驗重現論文 **Deep Region Adaptive Denoising for Texture Enhancement**（IEEE Access 2022，DOI `10.1109/ACCESS.2022.3222826`）前半段的 DCT texture classifier。目的不是重現完整 denoising network，而是先回答：

1. 依論文規則產生的 texture/non-texture pseudo labels 是否可學習？
2. 兩層 CNN 是否比直接 Sobel 或 DCT 高頻門檻更有價值？
3. 產生的 soft texture map 實際代表材質紋理，還是高頻細節？
4. 輸出能否直接和既有 TextureSAM masks 串接？

論文來源：

- https://doi.org/10.1109/ACCESS.2022.3222826
- https://pure.korea.ac.kr/en/publications/deep-region-adaptive-denoising-for-texture-enhancement/
- https://www.researchgate.net/publication/365483338_Deep_region_adaptive_denoising_for_texture_enhancement

## 2. 論文可確定的設定

- 以 Sobel gradient 分析 DIV2K。
- 每張影像最高梯度的 100×100 區域定義為 texture；最低者定義為 non-texture。
- 從這些區域隨機裁出 8×8 patches。
- 建立 3000 對 texture/non-texture patches。
- 每個 8×8 patch 先做 2-D DCT。
- classifier 包含兩個 3×3 convolution layers，各接 ReLU 與 max pooling，再接三個 fully-connected layers及 sigmoid。
- 全圖推論時，將每個 patch 的機率填滿其 8×8 視窗，再將所有重疊結果平均為逐像素 soft texture map。

## 3. 論文未交代而本實驗固定的重現假設

論文沒有完整提供 classifier 程式碼與以下細節，因此第一版明確固定為：

- 色彩輸入：OpenCV BT.601-style luminance，單通道。
- DCT：保留正負號的 orthonormal DCT-II。
- 正規化：以 training set 對每一個 DCT coefficient 計算 mean/std；validation 不參與統計。
- Conv channels：16、32；3×3、padding 1。
- FC widths：64、16、1。
- Loss：binary cross-entropy with logits。
- Optimizer：Adam，`lr=1e-4`、`betas=(0.9, 0.999)`。
- Batch size：128；最多 100 epochs；以 validation AUROC early stopping，patience 20。
- Dense inference：原始解析度、8×8、stride 1，不 resize；所有重疊預測精確平均。
- 固定 seed：`20260913`。

這些設定是重現假設，不宣稱為作者未公開的原始參數。

## 4. 資料

| Split | 來源影像 | 影像數 | Pair 數 | Patch 數 |
|---|---|---:|---:|---:|
| Train | DIV2K train HR | 800 | 3000 | 6000 |
| Validation | DIV2K validation HR | 100 | 1000 | 2000 |
| OOD qualitative | external grass | 4 | — | 全圖 |

Train 與 validation 按原始影像完全隔離，不會讓同一張影像的鄰近 patch 洩漏到兩個 split。

本實驗不需要人工 segmentation GT、預訓練模型或額外下載資料。Classifier 從頭訓練；監督訊號是 Sobel 規則產生的 pseudo labels。

## 5. Pseudo-label 統計與稽核

### 5.1 100×100 region 統計

| Split | 高梯度區平均 | 低梯度區平均 | 每張圖 high/low ratio 中位數 | high/low 有重疊的影像 |
|---|---:|---:|---:|---:|
| Train | 0.7643 | 0.02308 | 52.39× | 1 / 800 |
| Validation | 0.7289 | 0.02633 | 37.56× | 0 / 100 |

唯一重疊的 training image IoU 只有 0.00281，沒有造成大量相衝突取樣。

### 5.2 隨機 8×8 patch 統計

| Split | Texture Sobel mean | Non-texture Sobel mean | Texture DCT HF ratio | Non-texture DCT HF ratio |
|---|---:|---:|---:|---:|
| Train | 0.6258 | 0.01764 | 0.04739 | 0.01099 |
| Validation | 0.5775 | 0.02076 | 0.04582 | 0.01432 |

目視稽核確認，大部分 texture patches 含密集細節，但其中也包含強物體邊界、文字與直線；non-texture patches 多為天空、黑背景、牆面及模糊表面。這符合論文的操作性定義，但不是純材質標註。

稽核圖：`data/paper_v1/audit/train/`、`data/paper_v1/audit/val/`。

## 6. 模型與訓練結果

- GPU：NVIDIA GeForce RTX 4070 Ti SUPER
- PyTorch：2.5.1+cu124
- Trainable parameters：14,113
- 完成 epochs：83；最佳 checkpoint：epoch 63
- 記錄的訓練時間：約 8.26 秒

### 6.1 Patch-level quantitative result

所有 baseline threshold 只用 training set 選擇，然後鎖定到 validation。

| 方法 | Validation Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| DCT CNN | **0.9620** | **0.9617** | **0.9903** |
| 直接 Sobel patch mean | 0.9600 | 0.9592 | 0.9899 |
| 直接 DCT HF energy ratio | 0.8465 | 0.8555 | 0.8878 |

CNN 的 confusion matrix（threshold 0.5）：TP 953、TN 971、FP 29、FN 47。

### 6.2 CNN 是否只是重建 Sobel？

- CNN probability 與 patch Sobel score Pearson correlation：0.8364。
- CNN/Sobel binary decision agreement：98.5%。
- CNN 正確但 Sobel 錯誤：17 / 2000。
- Sobel 正確但 CNN 錯誤：13 / 2000。
- 兩者都錯：63 / 2000。

因此 CNN 相對直接 Sobel 只淨增加 4 個正確 patch，也就是 accuracy +0.2 percentage point。CNN 確實遠勝單一手工 DCT 高頻比例，但在 pseudo-label 自身的評估標準下，幾乎就是 Sobel 的非線性代理。

## 7. 原始解析度 soft texture map

對 8 張均勻抽樣 DIV2K validation images 與全部 4 張 external grass images，已完成 8×8、stride-1、無 resize 推論。每個像素平均所有覆蓋它的 patch probability。

### 7.1 External grass

| 影像 | 全圖平均 P(texture) | 像素 P≥0.5 比例 | 推論時間 |
|---|---:|---:|---:|
| pexels_golf_ball_grass_sky | 0.2477 | 22.7% | 2.05 s |
| piqsels_grass | 0.9982 | 100.0% | 0.24 s |
| unsplash_grass_3000 | 0.7271 | 92.0% | 2.35 s |
| wikimedia_golf_course | 0.3846 | 39.7% | 2.65 s |

觀察：

- 近距離、細節密集的草地會得到高機率。
- 有景深的高爾夫球場中，清晰草地高、模糊草地低，天空低。
- 遠景樹木、房屋、地平線與球體邊界也會被標高。
- 它反映「可見高頻細節」，不是「草地語意」或「整塊草材質」。

公開 Gallery：`results/paper_v1/external_grass_gallery.jpg`。

### 7.2 DIV2K qualitative

- 岩石、森林地面、細小花紋等區域通常高。
- 黑背景、天空、平滑物體表面通常低。
- 玫瑰花瓣邊緣、蝴蝶輪廓、蘑菇邊界等被強烈標高。
- 低對比但仍有視覺結構的表面常只得到中低機率。

公開 Gallery：`results/paper_v1/div2k_gallery.jpg`。

## 8. 受控失敗模式

所有影像皆為 512×512，使用相同 checkpoint 和 absolute probability；threshold 固定 0.5。

| 輸入 | 平均 P(texture) | P≥0.5 比例 | 判讀 |
|---|---:|---:|---|
| Constant flat | 0.0072 | 0.0% | 正確判低 |
| Single step edge | 0.0213 | 1.2% | 邊界帶被判成 texture |
| Smooth gradient | 0.0089 | 0.0% | 正確判低 |
| 4-pixel checkerboard | 1.0000 | 100.0% | 強規律紋理判高 |
| Low-contrast periodic texture | 0.0366 | 0.0% | 明顯漏掉弱紋理 |
| Gaussian noise, σ=15 | 0.9256 | 100.0% | 幾乎全部誤當 texture |
| Text and thin lines | 0.3330 | 33.0% | 線條及文字邊緣判高 |
| Texture contrast ramp | 0.0911 | 0.0% | 對紋理振幅非常敏感 |

這組結果直接證明：第一版 classifier 的主要能力是偵測「高對比、高頻能量」，而不是辨識紋理結構本身。它不應直接用於 noisy image；論文完整方法另外用 noisy-to-clean texture map generator 處理這個問題。

公開 Gallery：`results/paper_v1/diagnostics_gallery.jpg`。

## 9. TextureSAM 相容性

將四張 external grass 的 CNN soft map 與既有 TextureSAM 輸出串接：

- TextureSAM mask 數：11。
- Shape-compatible：11 / 11。
- Shape errors：0。
- CNN 輸出保留原始 H×W，不需 resize。
- 對每個 mask 計算 mean、median、P90、P≥0.5 fraction。
- 另輸出 16-bit `TextureSAM mask × CNN P(texture)` weighted mask。

例子：

- `piqsels_grass` mask 0：mean P=0.9983，mask 內 P≥0.5 為 100%。
- `unsplash_grass_3000` mask 0：mean P=0.7341，mask 內 P≥0.5 為 92.38%。
- `pexels` mask 1：mean P=0.0122；mask 0：mean P=0.3316，因此可用來排序或篩選 TextureSAM regions。
- `wikimedia_golf_course` mask 0：mean P=0.0177，顯示該 TextureSAM mask 本身並不是高紋理區；CNN 並不會因為它是 SAM mask 就自動判高。

公開摘要：`results/paper_v1/texturesam_compatibility_summary.json`。

這裡的整合方式是「TextureSAM 先提供區域，CNN 提供每像素 textureness，再做區域統計／加權」。目前 CNN 訓練本身沒有使用 TextureSAM pseudo labels，因為忠實的論文 baseline 使用 Sobel labels；若讓 TextureSAM 參與訓練，應另立改良實驗，不應混入第一版。

## 10. 第一版結論

### 成立

1. 論文描述的 pseudo-label → DCT CNN → dense soft map 流程可成功重現。
2. 模型小、訓練快，對論文 pseudo labels 有強泛化能力。
3. 原始解析度 stride-1 推論可執行，沒有 resize 導致細節消失。
4. 對 denoising 的「高頻細節保護圖」用途合理。
5. 與 TextureSAM 的輸出尺寸及資料型態可以直接串接。

### 不成立或尚未證明

1. 尚不能稱為一般材質紋理分類器。
2. 無法區分真實紋理和 Gaussian noise。
3. 會把 isolated edges、文字、輪廓視為 texture。
4. 會漏掉低對比規律紋理。
5. CNN 對 pseudo labels 的提升相對直接 Sobel 很小；96.2% 對 96.0%。
6. 評估 GT 仍來自 Sobel，本質上具有循環性，不能用 96.2% 宣稱人類紋理辨識正確率。

最準確的命名是 **DCT high-frequency detail probability map**。

## 11. 建議下一步（尚未執行）

如果目標仍是論文的 denoising，下一步應重現 map-generation subnet `G`，用 clean-image DCT map 當 pseudo GT，訓練 noisy image → clean texture map，尤其驗證 σ=15/25/50。

如果目標改為我們要的材質紋理圖，下一步不應直接擴大模型；應先建立一個獨立於 Sobel 的小型人工診斷集，至少標註：

- repeated/material texture
- isolated edge/text
- weak texture
- stochastic noise/artifact
- flat region

再測試 structure-tensor coherence、noise negatives、多尺度 receptive field 或 TextureSAM region context。否則只會繼續提高「模仿 Sobel」的分數。

## 12. 可重現產物

- 程式及命令：`README.md`
- DIV2K dataset 與衍生 patch data：不納入 repository，須由使用者自行下載/重建
- Best checkpoint：`checkpoints/paper_v1_dct_classifier.pt`
- Training history：`results/paper_v1/history.csv`
- Quantitative metrics：`results/paper_v1/metrics.json`
- CNN/Sobel analysis：`results/paper_v1/validation_analysis.json`
- Full-resolution selected galleries：`results/paper_v1/div2k_gallery.jpg`、`external_grass_gallery.jpg`
- Controlled diagnostics：`results/paper_v1/diagnostics_gallery.jpg`
- TextureSAM integration：`results/paper_v1/texturesam_compatibility_summary.json`
