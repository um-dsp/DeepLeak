import os
import pickle
import random
import numpy as np
import argparse
import pandas as pd
import optuna
import torch
from datasets.data_split import split_data
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import torchvision
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import MobileNetV2, MobileNetV2BCE
from utils.training import train_model
from utils.evaluation import evaluate_model
from datasets import DynamicGraphDataset,load_data
from xai_methods.generate_attributions import gen_attri
def run_attack_trial(
    trial,
    xai_method,
    seed=0,
    standard=False,
    args=""
):
    dataset_name = args.dataset_name
    model_path_shadow = args.model_path_shadow
    model_path_oracle = args.model_path_oracle
    data_root = args.data_root
    attri_root = args.attri_root
    save_model_root = args.save_model_root
    num_classes = args.num_classes
    attack_model_epochs = args.attack_model_epochs
    pickle_files=None
    # ------------------ PARAMETER SAMPLING ------------------
    if xai_method == "ProtoDa":
        batch_size = 256
        params = {"sigma": trial.suggest_float('sigma', 0.1, 3.0)}
    elif xai_method in ["SMAP", "GBackProp", "GGC", "GC", "DCAttr", "INGRAttr"]:
        batch_size = 128
        params = {
            "correc": not standard,
            "noise_level": trial.suggest_float('noise_level', 0.001, 0.2),
            "clamp_range": (
                trial.suggest_float('clamp_min', -5.0, 0.0),
                trial.suggest_float('clamp_max', 0.0, 5.0),
            ),
            "mask_threshold": trial.suggest_float('mask_threshold', 1e-4, 0.05),
        }
    elif xai_method in ["SmoothGrad", "VarGrad"]:
        batch_size = 3
        params = {
            "stdevs": trial.suggest_float("stdevs", 0.01, 2.0) if not standard else 1.0,
            "draw_baseline_from_distrib": trial.suggest_categorical("draw_baseline_from_distrib", [True, False]) if not standard else False
        }
    elif xai_method == "IG":
        batch_size = 16
        params = {
            "n_steps": trial.suggest_int('n_steps', 50, 50),
            "method": trial.suggest_categorical('method', ['gausslegendre']),
        }
    elif xai_method in ["SHAP", "LIME"]:
        batch_size = 512
        params = {
            "n_samples": trial.suggest_int('n_samples', 1, 25) if not standard else 25,
            "n_segments": trial.suggest_int('n_segments', 10, 100) if not standard else 50,
            "compactness": trial.suggest_int('compactness', 1, 50) if not standard else 10,
        }

    # ------------------ RANDOM SEED ------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------ LOAD DATA ------------------
    if dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    elif dataset_name == "GTSRB":
        raise NotImplementedError("Add GTSRB dataset logic")
    elif dataset_name == "DynamicGraph":
        trainset, testset = DynamicGraphDataset(split="train"), DynamicGraphDataset(split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # ------------------ LOAD PICKLES ------------------
    if pickle_files is None:
        pickle_files = {
            "subset1": os.path.join(data_root, f"train_subset_{dataset_name.lower()}.pickle"),
            "subset2": os.path.join(data_root, f"test_subset_{dataset_name.lower()}.pickle"),
            "subset3": os.path.join(data_root, f"disjointsubset_{dataset_name.lower()}.pickle"),
            "tries": os.path.join(data_root, f"best_index.pickle"),
            "oracle": os.path.join(data_root, f"balanced_{dataset_name.lower()}_train_indices.pkl")
        }

    subset1 = pickle.load(open(pickle_files["subset1"], 'rb'))
    subset2 = pickle.load(open(pickle_files["subset2"], 'rb'))
    subset3 = pickle.load(open(pickle_files["subset3"], 'rb'))
    random_indices_tries = pickle.load(open(pickle_files["tries"], 'rb'))
    random_indices_oracle = pickle.load(open(pickle_files["oracle"], 'rb'))

    # ------------------ MODELS ------------------
    if model_path_shadow is None:
        model_path_shadow = os.path.join(save_model_root, f"shadow_model_{dataset_name}BIS.pth")
    if model_path_oracle is None:
        model_path_oracle = os.path.join(save_model_root, f"OracleModel_{dataset_name}BIS.pth")

    model = MobileNetV2(num_classes)
    model.load_state_dict(torch.load(model_path_shadow))
    model.to(device)

    oracle_model = MobileNetV2(num_classes)
    oracle_model.load_state_dict(torch.load(model_path_oracle))
    oracle_model.to(device)

    # ------------------ SAMPLE SPLITS ------------------
    shadow_samples = [(subset1 + subset2)[i] for i in random_indices_tries]
    shadow_dt = [testset[i] for i in range(5000)]
    oracle_samples_test = [testset[i] for i in range(5000, 10000)]

    # ------------------ ATTRIBUTION GEN ------------------
    member_dir = os.path.join(attri_root, f"{dataset_name}/Member")
    nonmember_dir = os.path.join(attri_root, f"{dataset_name}/Non-Member")
    saveme_dir = os.path.join(attri_root, f"{dataset_name}/attri_saveme2")
    savenonme_dir = os.path.join(attri_root, f"{dataset_name}/attri_savenonme2")

    gen_attri(member_dir, shadow_samples, model, xai_method, target=1, batch_size=batch_size, metrics=False, **params)
    gen_attri(nonmember_dir, shadow_dt, model, xai_method, target=0, batch_size=batch_size, metrics=False, **params)

    dataset = load_data(member_dir)
    dataset_bis = load_data(nonmember_dir)
    combined_dataset = dataset + dataset_bis[:5000]

    # ------------------ ATTACK TRAIN ------------------
    train_size = int(0.8 * len(combined_dataset))
    trdasets, tedasets = random_split(combined_dataset, [train_size, len(combined_dataset) - train_size])
    train_dataset = [(torch.tensor(x[0]), torch.tensor(x[1])) for x in trdasets]
    test_dataset = [(torch.tensor(x[0]), torch.tensor(x[1])) for x in tedasets]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)

    attack_model = MobileNetV2BCE(1).to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    attack_epochs = attack_model_epochs or (15 if xai_method in ["Occlusion", "FeaPermAttr", "FeaAbAttr", "INGRAttr", "SHAP"] else 10)
    attack_model = train_model(attack_model, nn.BCELoss(), optimizer, train_loader, scheduler, test_loader, epochs=attack_epochs)

    torch.save(attack_model.state_dict(), os.path.join(save_model_root, f'AttackModel_{dataset_name}_{xai_method}.pth'))

    # ------------------ ORACLE ATTRIBUTION + EVAL ------------------
    balanced_trainset = Subset(trainset, random_indices_oracle)
    oracle_samples = [balanced_trainset[i] for i in range(5000)]

    sensitivity_me = gen_attri(saveme_dir, oracle_samples, oracle_model, xai_method, target=1, batch_size=batch_size, metrics=True, **params)
    sensitivity_nonme = gen_attri(savenonme_dir, oracle_samples_test, oracle_model, xai_method, target=0, batch_size=batch_size, metrics=True, **params)
    avg_sensitivity = (sensitivity_me + sensitivity_nonme) / 2

    final_dataset = load_data(saveme_dir ) + load_data(savenonme_dir)[:5000]
    final_dataset = [(torch.tensor(x[0]), torch.tensor(x[1])) for x in final_dataset]
    test_loader_final = DataLoader(final_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)

    roc_data_shadow = evaluate_model(attack_model, test_loader_final, verbose=True, roc_data_shadow=[], xai=xai_method)

    fpr_fixed = 0.001
    fprs = roc_data_shadow[0]['FPR']
    tprs = roc_data_shadow[0]['TPR']
    closest_index = (abs(fprs - fpr_fixed)).argmin()
    tpr_at_fpr = tprs[closest_index]

    print(f"TPR@FPR={fpr_fixed}: {tpr_at_fpr * 100:.2f}%, Avg Sensitivity: {avg_sensitivity:.4f}")
    return tpr_at_fpr * 100, avg_sensitivity




def run_attack_phase(xai_methods, topk, trials, csv_path,args):
    print(f" [Phase 1: Seed Selection] Evaluating seeds for XAI methods: {xai_methods}")
    results = []
    j = 0

    for xai in xai_methods:
        tprs = []
        seeds = []
        for i in range(trials):
            print(f" XAI: {xai} | Trial {i+1}/{trials}")
            study = optuna.create_study(directions=['minimize', 'minimize'])
            trial = study.ask()
            seed = j
            tpr_at_fpr, avg_sensitivity = run_attack_trial(trial, xai, seed, standard=True,args=args)
            print(f" Seed {seed} | TPR: {tpr_at_fpr:.4f}% | Sensitivity: {avg_sensitivity:.4f}")
            tprs.append(tpr_at_fpr)
            seeds.append(j)
            j += 1

        top_seeds = sorted(zip(tprs, seeds), key=lambda x: x[0], reverse=True)[:topk]
        results.extend([(xai, seed, tpr) for tpr, seed in top_seeds])

    df = pd.DataFrame(results, columns=['XAI Method', 'Seed', 'TPR'])
    df.to_csv(csv_path, index=False)
    print(f" Saved top-{topk} seeds per XAI method to {csv_path}")


def run_optimization_phase(trials, csv_path,args):
    print(f" [Phase 2: Optimization] Loading seeds from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run the attack phase first.")

    df = pd.read_csv(csv_path)
    xai_methods = df['XAI Method'].tolist()
    seed_xai = df['Seed'].tolist()
    studies = {}

    for xai, seed in zip(xai_methods, seed_xai):
        print(f"\n🚀 Optimizing XAI: {xai} | Seed: {seed}")
        study = optuna.create_study(directions=['minimize', 'minimize'])

        # First run baseline
        trial = study.ask()
        tpr_at_fpr, avg_sensitivity = run_attack_trial(trial, xai, seed, standard=True,args=args)
        print(f"Standard Seed Result — TPR: {tpr_at_fpr:.4f}% | Sensitivity: {avg_sensitivity:.4f}")
        study.tell(trial, [tpr_at_fpr, avg_sensitivity])

        # Then optimize
        study.optimize(lambda trial: run_attack_trial(trial, xai, seed, standard=False,args=args), n_trials=trials)

        pareto_front = study.best_trials
        for trial in pareto_front:
            print(f"\n Pareto Trial {trial.number}:")
            print(f"TPR: {trial.values[0]:.4f}% | Sensitivity: {trial.values[1]:.4f}")
            print("Parameters:", trial.params)

        with open(f'optuna_results_{xai}+{seed}.pkl', 'wb') as f:
            pickle.dump({'study': study, 'pareto_front': pareto_front}, f)

        studies[xai] = study

    return studies


import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="XAI Attack + Optimization Pipeline")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["attack", "optimize", "both"],
                        help="Phase to run")
    parser.add_argument("--xai_methods", type=str, nargs='+',
                        help="List of XAI methods (required for attack or both modes)")
    parser.add_argument("--topk", type=int, default=3,
                        help="Top-K seeds to select in attack phase")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials per method")
    parser.add_argument("--csv_path", type=str, default="top_3_tprs.csv",
                        help="CSV path to store/load seeds")

    # General config
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--model_path_shadow", type=str, default=None)
    parser.add_argument("--model_path_oracle", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--attri_root", type=str, default="./data/attributions")
    parser.add_argument("--save_model_root", type=str, default="./models")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--attack_model_epochs", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(f"data/disjointsubset_{args.dataset_name}.pickle"):
        split_data()

    os.makedirs("./data/attributions_"+args.dataset_name+"/Members", exist_ok=True)
    os.makedirs("./data/attributions_"+args.dataset_name+"/Non-Members", exist_ok=True)
    os.makedirs("./data/attributions_"+args.dataset_name+"/attri_saveme2", exist_ok=True)
    os.makedirs("./data/attributions_"+args.dataset_name+"/attri_savenonme2", exist_ok=True)
    if args.mode == "attack":
        if not args.xai_methods:
            raise ValueError("--xai_methods is required when mode is 'attack'")
        run_attack_phase(args.xai_methods, args.topk, args.trials, args.csv_path,args)

    elif args.mode == "optimize":
        run_optimization_phase(args.trials, args.csv_path,args)

    elif args.mode == "both":
        if not args.xai_methods:
            raise ValueError("--xai_methods is required when mode is 'both'")
        run_attack_phase(args.xai_methods, args.topk, args.trials, args.csv_path,args)
        run_optimization_phase(args.trials, args.csv_path,args)
