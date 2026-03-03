RUN_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="Logs/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
python -u train.py --cuda 0 --lr 0.001 --eval_freq 25 --dim 200 --dataset DB15K-tuning --epochs 500 --lamda_l 1e-5 --lamda_g 1e-5 > "${LOG_DIR}/DB15K-tuning_g-1e-5_l-1e-5.txt" 2>&1;
python -u train.py --cuda 0 --lr 0.001 --eval_freq 25 --dim 200 --dataset MKG-W-tuning --epochs 500 --lamda_l 5e-5 --lamda_g 1e-4 > "${LOG_DIR}/MKG-W-tuning_g-1e-4_l-5e-5.txt" 2>&1;
python -u train.py --cuda 0 --lr 0.001 --eval_freq 25 --dim 200 --dataset MKG-Y-tuning --epochs 500 --lamda_l 5e-4 --lamda_g 1e-6 > "${LOG_DIR}/MKG-Y-tuning_g-1e-6_l-5e-4.txt" 2>&1;
