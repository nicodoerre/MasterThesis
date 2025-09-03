# --------------------------------------------------------------------
# search_hyper_params_wavemix.py
import subprocess, sys, optuna, pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent          # folder …
TRAIN_PY  = REPO_ROOT / "train_div2k.py"                    # WaveMixSR
DIR_DATA  = r"C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation"
SCALE     = 4                                              # 2 / 3 / 4
EPOCHS    = 10

def objective(trial):
    # only two hyper-parameters
    bs = trial.suggest_categorical("batch_size", [4, 8, 16])
    lr_adam = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    lr_sgd = trial.suggest_float("lr_sgd", 1e-5, 5e-4, log=True)

    #run_name = f"wavemix_lr{lr:.1e}_bs{bs}_trial{trial.number}"

    cli = [
        sys.executable, str(TRAIN_PY),
        "-x", str(SCALE),                 #  --resolution
        "--bs", str(bs),
        "--lr_adam", str(lr_adam),
        "--lr_sgd", str(lr_sgd)
    ]


    # run training
    proc = subprocess.run(cli, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)            # show traceback once
        raise optuna.TrialPruned()

    # parse final PSNR from stdout
    psnr = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PSNR_y:"):              # printed by train_div2k.py
            psnr = float(line.split()[1])
            break
    if psnr is None:
        raise optuna.TrialPruned()

    return psnr

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best value:", study.best_value)
print("Best params", study.best_params)
# --------------------------------------------------------------------


#for normal training call: python train_div2k.py -x 2 -metric psnr --lr_adam 4.171034750996214e-05 --lr_sgd 1.5989466860124584e-05 --bs 4
    #                      python train_div2k.py -x 3 -metric psnr --lr_adam 4.652145176766172e-05 --lr_sgd  3.118558640933793e-05 --bs 4
    #                      python train_div2k.py -x 4 -metric psnr --lr_adam  0.00020073916171442198 --lr_sgd 8.099326645983506e-05 --bs 4 
    
# for x2: 'batch_size': 4, 'lr': 4.171034750996214e-05, 'lr_sgd': 1.5989466860124584e-05
# for x3: 'batch_size': 4, 'lr': 4.652145176766172e-05, 'lr_sgd': 3.118558640933793e-05
# for x4: 'batch_size': 4, 'lr': 0.00020073916171442198, 'lr_sgd': 8.099326645983506e-05