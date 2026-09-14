# 抗噪紋理分類器與全圖紋理圖生成器：第一輪完整實驗

實驗日期：2026-09-13  
專案：`DCT-Texture-Classifier`  
狀態：A–E 實驗、控制組、原圖推論、TextureSAM 介面驗證均已完成

## 1. 結論先講

1. **DCT 不是不能做抗噪紋理判斷，但 DCT 本身不會把 noise 與真實紋理分開。**乾淨影像訓練的 DCT 分類器遇到 Gaussian noise sigma=15 時，accuracy 從 96.20% 掉到 51.80%，幾乎把所有 patch 都叫做紋理。
2. **讓 DCT 分類器看 noisy patch 確實有效。**混合 sigma 0–50 訓練後，sigma=15 accuracy 為 92.05%，sigma=50 為 73.95%。所以問題並非理論上做不到，而是模型需要從 paired clean/noisy 資料學會抗噪。
3. **同樣大小的模型，直接看 spatial 8×8 比看 DCT 8×8 更耐強噪聲。**兩者都是 14,113 parameters；sigma=50 分別為 83.45% 與 73.95%。DCT 在這個任務中不是必要條件。
4. **真正最有效的是較大的 spatial context。**32×32 spatial CNN 在 sigma=50 仍有 96.85%，顯著高於 8×8 DCT/Spatial。模型能利用較大範圍判斷「有組織、可延續的結構」或「隨機、局部不一致的噪聲」。
5. **Spatial 32×32 + 中央 8×8 DCT 的 hybrid 並沒有穩定勝出。**sigma 0–35 略高，但 sigma=50/60 反而低於純 spatial 32×32。因此目前沒有證據支持保留 DCT branch。
6. **全圖模型 G 能從 noisy image 直接產生接近 clean teacher 的 dense texture map。**在 DIV2K validation crops、sigma=50 時，G 的 binary map accuracy 為 83.48%，直接把 noisy image 丟給乾淨 DCT teacher 只有 58.76%，且後者幾乎整張變成 texture。
7. **G 不是普適紋理規則。**它在 held-out 自然影像上有效，但人工 checkerboard 被整張漏掉，低對比週期圖卻幾乎整張誤報。它學到的是 DIV2K 自然影像先驗與 teacher 的關係，而不是可靠的頻率公理。
8. **新 G 可以直接使用 TextureSAM 輸出。**4 張原尺寸草地圖均未 resize；11/11 個既有 TextureSAM masks 形狀完全相容，已產生 weighted masks。

所以，對「可抗 noise 的紋理圖」這個方向，第一候選應該是 **raw spatial、較大 receptive field、paired clean/noisy training**。DCT 可作為輔助特徵或可解釋的 baseline，不應預設為必要輸入。

## 2. 本輪回答的核心問題

原始問題是：DCT 是否使 noise 與真實紋理無法區分？是否可以不用 DCT？

DCT 是線性基底轉換。白噪聲會把能量廣泛灑到高頻係數；邊緣、細線與週期紋理也會在高頻係數產生能量。單看「高頻能量大」時，兩者確實不可識別。可是 DCT 還保留了係數的空間頻率位置與符號，學習器仍可能利用其統計排列區分部分結構與噪聲。因此應由對照實驗回答，而不是由 DCT 名稱推論。

本輪結果顯示：

- clean-only DCT 的確把 noise 當 texture；
- noise-trained DCT 能學到一部分區分能力；
- raw spatial 在相同 parameter budget 下更好；
- 增加 spatial context 帶來最大提升；
- 已知 sigma 可救回 DCT 的高噪聲結果，但仍不及 32×32 spatial context。

換言之：**可以不用 DCT，而且本輪實驗支持不用 DCT 作為主要表示。**

## 3. 資料、GT 與沒有做的 resize

### Patch screening A–D

- Train：DIV2K train 800 張影像，6,000 patches（3,000 texture + 3,000 non-texture）。
- Validation：獨立 DIV2K validation 100 張影像，2,000 patches（1,000 + 1,000）。
- Label：由 clean image 的 Sobel magnitude 選出每張圖最高/最低的 100×100 區域，再從中取 8×8 patch。這是 pseudo-GT，不是人工材質標註。
- 32×32 context：以同一個 8×8 patch 為中心，由原圖取周圍像素；6,000/2,000 個中心 patch 全部與 v1 完全一致。
- Noise：AWGN，8-bit sigma 0–60；mixed training 是每個 sample 隨機 Uniform[0,50]。
- **沒有把整張影像 resize。**所有 patch/context 都從原始解析度裁切，避免細紋理被縮圖抹掉。

### Dense map generator E

- Train：DIV2K train 的 1,600 個 128×128 native-resolution random crops，每張原圖 2 crops。
- Validation：獨立 DIV2K validation 的 200 crops，每張原圖 2 crops。
- Input：由 clean crop 即時加入 AWGN，sigma 每個 crop 隨機 Uniform[0,50]。
- Output：同尺寸 128×128 soft texture probability map。
- Target：v1 clean DCT classifier 在 clean crop 上 stride=1 推論，再把所有 overlapping 8×8 probabilities 精確平均所得的 map。
- Loss：pixel MSE。
- GT 本質：teacher distillation target，仍不是人類唯一正解。

## 4. 實驗矩陣

| ID | 模型 | Input | Parameters | 訓練噪聲 | Output |
|---|---|---:|---:|---|---|
| A0 | v1 clean DCT | noisy central 8×8 的 DCT | 14,113 | clean only | binary patch probability |
| A1–A3 | fixed DCT | noisy central 8×8 的 DCT | 14,113 | fixed 15 / 25 / 50 | binary patch probability |
| A4 | mixed DCT | noisy central 8×8 的 DCT | 14,113 | U[0,50] | binary patch probability |
| A5 | conditional DCT | DCT + 已知 sigma | 14,177 | U[0,50] | binary patch probability |
| B | spatial 8 | raw noisy central 8×8 | 14,113 | U[0,50] | binary patch probability |
| C | spatial 32 | raw noisy 32×32 context | 39,777 | U[0,50] | central patch probability |
| D | hybrid 32 | raw 32×32 + central 8×8 DCT | 52,769 | U[0,50] | central patch probability |
| E | map generator G | raw noisy 128×128 crop | 761,665 | U[0,50] | dense 128×128 soft map |

Patch CNN 均用 Adam、BCE；G 用 Adam、pixel MSE。固定驗證 noise seed，所有模型在相同 validation patches 上比較。Conditional DCT 最終 checkpoint 以多個 sigma 的平均 validation accuracy 選取；第一次以 AUROC 選取會造成 threshold calibration 很差，因此保留原始紀錄但不拿它當最終結論。

## 5. Patch 實驗結果

Accuracy，threshold 固定 0.5：

| 模型 | σ0 | σ5 | σ15 | σ25 | σ35 | σ50 | σ60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DCT clean-only v1 | .9620 | .8780 | .5180 | .5040 | .5005 | .5000 | .5000 |
| DCT fixed15 | .9300 | .9310 | .9390 | .6450 | .5255 | .5010 | .5000 |
| DCT fixed25 | .8810 | .8840 | .9030 | .9205 | .7405 | .5585 | .5145 |
| DCT fixed50 | .7395 | .7390 | .7570 | .7930 | .8375 | .8575 | .7895 |
| DCT mixed | .9265 | .9215 | .9205 | .9075 | .8455 | .7395 | .6810 |
| DCT mixed + sigma | .9490 | .9390 | .9255 | .9180 | .8885 | .8355 | .8020 |
| Spatial 8 mixed | .9360 | .9345 | .9225 | .9010 | .8815 | .8345 | .7920 |
| Spatial 32 mixed | .9795 | .9800 | .9795 | .9775 | .9760 | **.9685** | **.9460** |
| Hybrid 32 mixed | **.9815** | **.9815** | **.9810** | **.9810** | **.9790** | .9410 | .9095 |
| G 的中央 8×8 平均分數* | .9610 | .9615 | .9635 | .9670 | .9700 | .9410 | .8850 |

`*` G 是 dense map 模型且訓練目標不同，這列只是用相同 extreme-patch labels 做額外 probe，不能視為與 patch CNN 完全公平的架構排名。

觀察：

- Fixed-sigma model 只在接近訓練 sigma 時可靠，domain mismatch 很嚴重。
- Mixed noise 是合理的預設。
- 把 sigma 明確提供給 DCT，可將 sigma=50 從 73.95% 提升至 83.55%；顯示 DCT 表示中的歧義可由 noise-level prior 部分解除。
- Spatial 8 與 conditional DCT 在 sigma=50 幾乎相同，但 spatial 8 不需要先知道 sigma。
- Spatial 32 的優勢非常大。不過 labels 取自 100×100 極高/極低 Sobel 區，因此 32×32 周圍通常同樣屬於高/低細節區；這是設計要測的 context 效果，也讓任務比任意邊界 patch 容易。此結果不可直接外推為通用 pixel segmentation accuracy。

## 6. Dense map generator G

G 是 residual U-Net-like model：三次 down/up，加上 full-resolution skip；共 761,665 parameters。最佳 checkpoint 在 epoch 38，40 epochs 全部訓練與完整評估約 49.2 秒（CUDA）。

與 clean teacher target 比較：

| σ | G RMSE | G Pearson | G binary acc. | noisy image 直接進 clean DCT：RMSE | Direct binary acc. |
|---:|---:|---:|---:|---:|---:|
| 0 | .2095 | .8602 | .8624 | .0000 | 1.0000 |
| 5 | .2091 | .8603 | .8627 | .2011 | .8900 |
| 15 | .2077 | .8585 | **.8626** | .5292 | .5878 |
| 25 | .2092 | .8513 | **.8607** | .5716 | .5876 |
| 35 | .2183 | .8352 | **.8488** | .5786 | .5876 |
| 50 | .2411 | .8105 | **.8348** | .5810 | .5876 |
| 60 | .2853 | .7592 | **.7873** | .5814 | .5876 |

Direct clean-DCT 從 sigma=15 起 prediction mean 已達 .9569，等於幾乎全圖判成紋理；sigma=50 達 .9989。G 在訓練範圍內保留了空間結構，而且對未在訓練範圍內的 sigma=60 仍有部分泛化。

但 clean input 上，teacher 自己當然是完美 reference，而 G 只有 .8624 binary accuracy。這是 student approximation、MSE 平滑與有限資料造成的 fidelity loss。因此若輸入確定乾淨，沒有理由用 G 取代 teacher；G 的價值是 noisy input。

## 7. Controlled diagnostics：成功與失敗都重要

以 clean teacher target 為 reference：

- Flat：target 全低。sigma=15 時 direct DCT 平均 .9265，G 只有 .0855 且 0% pixels 超過 .5；成功排除 noise。
- Single edge：sigma=15 時 G binary accuracy .9924；保住窄 edge，而不是整張報 texture。
- Smooth gradient：sigma=50 時 G 仍有 .9844 binary accuracy；大部分保持 non-texture。
- Text/lines：sigma=15 時 G accuracy .9339、Pearson .9619；能恢復結構。
- **4-pixel checkerboard：teacher target 為 100% texture，但 G 即使 clean input 也判成 0% texture。完全失敗。**
- **Low-contrast periodic texture：teacher target 為 0% high，但 G clean input 判成 99.93% high。與 teacher 幾乎相反。**
- Texture contrast ramp：G 能看出對比的空間變化，但絕對分數比 teacher 高很多，0.5 threshold agreement 差。

這些結果否定了「validation 指標高，所以已學到通用紋理定義」的說法。它目前只是一個有效的 natural-image teacher imitator，而且 OOD synthetic pattern 仍可能災難性失敗。

## 8. TextureSAM 相容性

G 新增了 full-resolution tiled inference：

- 原圖不 resize；
- tile 128×128、overlap 64、外圍 reflect halo 32；
- 用有下限的 2-D Hann window overlap-add，減少 tile seam；
- 4 張 external grass test 共約 2.63 秒；
- 每張輸出與原圖相同 H×W 的 float32 `texture_probability.npy`。

用這些 maps 讀取既有 TextureSAM outputs：

- compatible masks：11/11；
- shape mismatch：0；
- missing/unreadable：0；
- 每個 mask 已輸出 `weighted_mask_*.png` 與 mean/median/P90 texture probability。

這證明資料介面成立：TextureSAM 提供區域，G 提供每個 pixel 的 texture confidence，兩者可以做 mask 內統計或 soft weighting。它不等於兩個模型已 joint-train，也不替 TextureSAM 的區域品質背書。

### 8.1 Freq-Aware-Seg demo 原圖追加測試

另以 `Freq-Aware-Seg/docs/demo_results/originals` 的完整 5 張 demo 原圖測試 G，全部維持原解析度，分別加入 sigma 0、15、50 AWGN。高紋理像素比例（P >= 0.5）如下：

| Demo image | sigma 0 | sigma 15 | sigma 50 |
|---|---:|---:|---:|
| div2k_0524 | 27.3% | 29.3% | 44.3% |
| pexels_golf_ball_grass_sky | 35.6% | 37.7% | 46.2% |
| piqsels_grass | 0.9% | 1.1% | 17.6% |
| unsplash_grass_3000 | 43.0% | 45.8% | 58.5% |
| wikimedia_golf_course | 35.1% | 36.1% | 44.1% |

sigma 15 的紋理空間結構與 clean input 接近；sigma 50 仍保留主要物體/邊緣結構，但所有影像的高紋理比例均上升，顯示強噪聲殘留被誤認為紋理。`piqsels_grass` 的 clean map 原本幾乎全低，在 sigma 50 增至 17.6%，是最清楚的 false-positive stress case。公開版合併比較圖為 `results/robust_v2/freq_aware_seg_demo_noise_comparison.jpg`。

## 9. Spatial 32×32 full-image diagnostic

為檢查高 patch accuracy 是否能直接轉成 dense map，另將 Spatial 32×32
classifier 以不重疊 8×8 target blocks 跑過 `Freq-Aware-Seg` 的五張完整
demo 原圖。每個 block 使用完整 32×32 reflected context，保持原生解析度，
並比較 sigma 0、15、50。

標準 threshold 0.5 下，各圖被判為 texture 的比例只有 0–1.3%。sigma 15
相對 sigma 0 的平均 binary agreement 為 99.94%，sigma 50 為 99.76%；但這些
數字受到大量 negative predictions 膨脹。視覺上高分結構在 sigma 15 大致
穩定，sigma 50 仍可辨認主要輪廓，然而絕對機率過低。

這項結果顯示：97% patch accuracy 是在每張圖極高／極低 Sobel 區域抽出的
平衡 validation patches 上成立。任意 full-image 位置包含大量未參與訓練的
中間難度區域，因此不能把該數字解讀為 dense-map accuracy，也不應直接用
0.5 threshold 部署。結果位於
`results/robust_v2/spatial32_demo_noise_comparison.jpg/.json`。

## 10. 本輪限制

1. GT 是 Sobel extreme pseudo-label 與其訓練出的 DCT teacher，不是人類材質標註；所有 accuracy 應解讀為「重現這個操作型定義」。
2. 只有單一主要 random seed；足以作第一輪 PoC 與大幅差異判斷，不足以對 0.1–1% 的微小差異下定論。
3. Noise 只有 clipped AWGN；尚未測 Poisson、shot/read noise、demosaic、JPEG、blur 或真實相機 noise。
4. Patch labels 是每張圖的極端區域，不能代表所有模糊邊界與中間難度 pixels。
5. G 只看 DIV2K natural images，synthetic OOD failures 已證明資料涵蓋不足。
6. Texture map 是「細節/高頻是否值得保留」的 denoising 輔助圖，不是相同材質的 instance/semantic clustering。

## 11. 建議下一步

下一輪不應先擴大模型，而應先修正學習目標：

1. 把人工 flat、edge、checker、不同頻率/對比週期圖加入 train/validation，特別修補本輪兩個災難性 case。
2. 加入多種 noise 與 paired clean/noisy real camera data；target 一律從 clean signal 產生。
3. 將 hard high/low extremes 改成 dense soft target，再加入 edge-vs-repetitive-texture 的可控標籤，清楚決定 edge 是否算 texture。
4. 以 Spatial 32 或全圖 raw-spatial G 為主幹；DCT 只作 optional auxiliary branch，做 multi-seed ablation 後才決定是否保留。
5. 把 TextureSAM masks 當區域約束或統計單位，而不是 texture GT；避免模型只學 TextureSAM 的錯誤。

## 12. 產物位置

- 綜合機器可讀摘要：`results/robust_v2/summary.json`
- Accuracy 曲線：`results/robust_v2/accuracy_vs_noise.png`
- Patch 全部原始 metrics：`results/robust_v2/patch_models_summary.json`
- Conditional DCT 公平重跑：`results/robust_v2/conditional_accuracy_selected_summary.json`
- G checkpoint：`checkpoints/map_generator_g.pt`
- G dense metrics：`results/robust_v2/metrics.json`
- Natural validation 視覺：`results/robust_v2/validation_sigma_15.jpg`、`validation_sigma_50.jpg`
- Controlled diagnostic CSV：`results/robust_v2/controlled_diagnostics.csv`
- Controlled diagnostic galleries：`results/robust_v2/g_diagnostics_sigma15_gallery.jpg`、`g_diagnostics_sigma50_gallery.jpg`
- External grass 公開結果：`results/robust_v2/map_generator_external_grass_gallery.jpg`
- TextureSAM 相容性摘要：`results/robust_v2/texturesam_compatibility_summary.json`

## 13. 重現指令

```powershell
$python = 'python'
$train = 'C:\path\to\DIV2K_train_HR'
$val = 'C:\path\to\DIV2K_valid_HR'

& $python build_dataset.py --train-dir $train --val-dir $val
& $python train.py

& $python build_context_dataset.py --train-images $train --val-images $val
& $python train_robust_patch.py

& $python build_map_dataset.py --train-images $train --val-images $val
& $python train_map_generator.py
& $python evaluate_g_on_patch_labels.py
& $python evaluate_g_diagnostics.py --input-dir assets\diagnostics_inputs
& $python visualize_map_crops.py

& $python infer_map_generator.py `
  --input-dir C:\path\to\test_images `
  --output-dir outputs\robust_v2\map_generator_external_grass

& $python score_texturesam_masks.py `
  --probability-root outputs\robust_v2\map_generator_external_grass `
  --mask-dir C:\path\to\texturesam_masks `
  --output-dir outputs\robust_v2\map_generator_texturesam_compatibility

& $python summarize_robust_v2.py
& $python -m unittest discover -s tests -v
```
