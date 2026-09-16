@echo off
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
> "logs\best_p_external_suite.status.txt" echo running
"C:\Users\LeoTsai\anaconda3\envs\texturesam\python.exe" infer_best_p_external_suite.py ^
  --input-dir "..\Freq-Aware-Seg\data\external_grass_test" ^
  --checkpoint "checkpoints\stcnn_limit\stcnn_diverse_conditioned_oracle.pt" ^
  --output-root "outputs\stcnn_stride1\external_grass_best_p_suite" ^
  --result-root "results\stcnn_stride1\external_grass_best_p_suite" ^
  --sigmas 0 15 50 ^
  --batch-size 8192 ^
  1> "logs\best_p_external_suite.stdout.log" ^
  2> "logs\best_p_external_suite.stderr.log"
set "RUN_RESULT=%ERRORLEVEL%"
> "logs\best_p_external_suite.status.txt" echo finished exit_code=%RUN_RESULT%
exit /b %RUN_RESULT%
