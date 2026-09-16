@echo off
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
> "logs\robust_p_external_suite.status.txt" echo running
"C:\Users\LeoTsai\anaconda3\envs\texturesam\python.exe" infer_best_p_external_suite.py ^
  --input-dir "..\Freq-Aware-Seg\data\external_grass_test" ^
  --checkpoint "checkpoints\stcnn_limit\stcnn_diverse_teacher.pt" ^
  --output-root "outputs\stcnn_stride1\external_grass_robust_p_suite" ^
  --result-root "results\stcnn_stride1\external_grass_robust_p_suite" ^
  --sigmas 0 15 50 ^
  --batch-size 4096 ^
  --model-label "P-robust (diverse large, no sigma input)" ^
  1> "logs\robust_p_external_suite.stdout.log" ^
  2> "logs\robust_p_external_suite.stderr.log"
set "RUN_RESULT=%ERRORLEVEL%"
if not "%RUN_RESULT%"=="0" goto finished
"C:\Users\LeoTsai\anaconda3\envs\texturesam\python.exe" make_external_suite_gallery.py ^
  --input-dir "results\stcnn_stride1\external_grass_robust_p_suite" ^
  --output "results\stcnn_stride1\external_grass_robust_p_suite_all.jpg" ^
  --width 3000 ^
  1>> "logs\robust_p_external_suite.stdout.log" ^
  2>> "logs\robust_p_external_suite.stderr.log"
set "RUN_RESULT=%ERRORLEVEL%"
:finished
> "logs\robust_p_external_suite.status.txt" echo finished exit_code=%RUN_RESULT%
exit /b %RUN_RESULT%
