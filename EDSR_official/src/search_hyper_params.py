import subprocess, sys, optuna, itertools, pathlib, shutil

REPO_ROOT = pathlib.Path(__file__).resolve().parent        # folder of search_hyper_params.py
MAIN_PY   = REPO_ROOT / "main.py"
DIR_DATA  = r"path/to/your/dataset"  # path to the folder containing the dataset folders (e.g. DIV2K, DIAGRAMS, ...)
SCALE     = 4
EPOCHS    = 10

BEST_BS = 16
BEST_LR = 0.00010978467846601387 # for x2: 0.00010978467846601387, x3: 6.759822466845348e-05, x4: 5.180994797434419e-05

def objective(trial):
    #bs    = trial.suggest_categorical("bs", [8, 16])
    #lr    = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    
    alpha = trial.suggest_float('alpha', 0.5, 2.0, log=True)
    beta  = trial.suggest_float('beta',  0.01, 0.5, log=True)
    gamma = trial.suggest_float('gamma_loss', 0.01, 0.5, log=True)
    delta = trial.suggest_float('delta', 0.01, 0.5, log=True)

    run_name = f"trial_{trial.number}"

    cli = [
        sys.executable, str(MAIN_PY),
        "--template","EDSR_paper",
        "--scale", str(SCALE),
        "--patch_size", str(96), #96,144,192
        "--n_resblocks", str(32),
        "--n_feats", str(256),
        "--batch_size", str(BEST_BS),
        "--lr", str(BEST_LR),
        "--res_scale", str(0.1),
        "--epochs", str(EPOCHS),
        "--test_every","50",
        "--data_train","DIAGRAMS_MINI",
        "--data_test","DIAGRAMS_MINI",
        "--loss", "1*CUSTOM", #only if training with custom loss
        "--alpha", str(alpha),
        "--beta",  str(beta),
        "--gamma_loss", str(gamma),
        "--delta", str(delta),
        "--dir_data", DIR_DATA,
        "--ext","img",
        "--n_threads","0",              # no worker processes
        "--save", run_name,
        "--reset"
    ]

    # run the training script
    result = subprocess.run(cli, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout)            # show error once
        raise optuna.TrialPruned()

    # read PSNR from the end of log.txt
    log_path = pathlib.Path("path/to/your/logs", run_name, "log.txt")
    for line in reversed(log_path.read_text().splitlines()):
        if "PSNR:" in line:
            return float(line.split("PSNR:")[1].split()[0])

    raise optuna.TrialPruned()          # if PSNR not found

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best:", study.best_value, study.best_params)

# for x2: Best: 13.619 {'bs': 16, 'lr': 0.00010978467846601387}
# call python main.py --template EDSR_paper --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --test_every 300 --batch_size 16 --lr 0.00010978467846601387 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS --data_test DIAGRAMS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep --save edsr_x2_mydataset_hyperparams --reset

# for x3: Best: 20.082 {'bs': 16, 'lr': 6.759822466845348e-05}
# call python main.py --template EDSR_paper --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --test_every 300 --batch_size 16 --lr 6.759822466845348e-05 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS --data_test DIAGRAMS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train path/to/your/pretrained/model --ext sep --save path/to/your/save/directory --reset

# for x4: 19.241 {'bs': 16, 'lr': 5.180994797434419e-05}
# call python main.py --template EDSR_paper --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --test_every 300 --batch_size 16 --lr 5.180994797434419e-05 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS --data_test DIAGRAMS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train path/to/your/pretrained/model --ext sep --save path/to/your/save/directory --reset

'''
x2:
done python main.py --template EDSR --scale 2 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep_reset --test_only --save init_cache
done python main.py --template EDSR_paper --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --test_every 21 --loss 1*CUSTOM --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep --save edsr_x2_mydataset_custom --reset
done python main.py --template EDSR_paper --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --test_every 21  --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep --save edsr_x2_mydataset_custom_plain --reset

x3:
done python main.py --template EDSR --scale 3 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep_reset --test_only --save init_cache
python main.py --template EDSR_paper --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --test_every 21 --loss 1*CUSTOM --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/EDSR_official/experiment/edsr_x2_mydataset_custom/model/model_best.pt --ext sep --save edsr_x3_mydataset_custom --reset
python main.py --template EDSR_paper --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --test_every 21  --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/EDSR_official/experiment/edsr_x2_mydataset_custom_plain/model/model_best.pt --ext sep --save edsr_x3_mydataset_custom_plain --reset

python main.py --template EDSR --scale 4 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --ext sep_reset --test_only --save init_cache
python main.py --template EDSR_paper --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --test_every 21 --loss 1*CUSTOM --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/EDSR_official/experiment/edsr_x2_mydataset_custom/model/model_best.pt --ext sep --save edsr_x4_mydataset_custom --reset
python main.py --template EDSR_paper --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --test_every 21  --print_every 5 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS_CUSTOM_LOSS --data_test DIAGRAMS_CUSTOM_LOSS --dir_data C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation --pre_train C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/EDSR_official/experiment/edsr_x2_mydataset_custom_plain/model/model_best.pt --ext sep --save edsr_x4_mydataset_custom_plain --reset
'''