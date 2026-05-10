# python ./co3d-main/co3d/download_dataset.py --download_folder co3d --single_sequence_subset --n_download_workers 4
python download_objaverse.py

## 特别提醒：如果c盘不够，请先运行 
# cmd /c mklink /J "%USERPROFILE%\.objaverse" YOUR_PATH_TO_DATASET