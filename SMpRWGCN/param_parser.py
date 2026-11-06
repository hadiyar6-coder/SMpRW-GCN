"""Parsing the parameters."""

import argparse
import os
import copy
from tabulate import tabulate


# ==========================
# bool converter
# ==========================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("true", "1", "yes", "y", "t"):
        return True
    if str(v).lower() in ("false", "0", "no", "n", "f"):
        return False
    return False


# ==========================
#  print args
# ==========================
def print_args_as_table(args, title="ARGS (Final Used)"):
    args_dict = vars(args)
    table = [[key, value] for key, value in args_dict.items()]
    print("\n====================", title, "====================")
    print(tabulate(table, headers=["Parameter", "Value"]))
    print("=========================================================\n")


# ==========================
# read all rows from config txt
# ==========================
def para_load(path):
    args_runs = []
    args_runs_element = {}

    if not os.path.exists(path):
        print(f"[WARNING] Config file not found: {path}")
        return []

    with open(path, encoding="utf-8") as para_file:
        first_line = para_file.readline().strip()
        dict_key = first_line.split('\t')

        for line in para_file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue

            parts = line.split('\t')
            if len(parts) != len(dict_key):
                print("[WARNING] Bad config line:", line)
                continue

            for i in range(len(parts)):
                args_runs_element[dict_key[i]] = parts[i]

            args_runs.append(copy.deepcopy(args_runs_element))

    return args_runs


# ==========================
#  update args using selected row
# ==========================
def para_update(args, row):
    keys = list(row.keys())

    # normalize alpha beta gama if exists
    if 'alpha' in row and 'beta' in row and 'gama' in row:
        try:
            s = float(row['alpha']) + float(row['beta']) + float(row['gama'])
            if s > 0:
                row['alpha'] = float(row['alpha']) / s
                row['beta'] = float(row['beta']) / s
                row['gama'] = float(row['gama']) / s
        except:
            pass

    for key in keys:
        if key in vars(args):
            expected_type = type(vars(args)[key])

            try:
                if expected_type == bool:
                    vars(args)[key] = str2bool(row[key])
                elif expected_type == int:
                    vars(args)[key] = int(float(row[key]))
                elif expected_type == float:
                    vars(args)[key] = float(row[key])
                else:
                    vars(args)[key] = expected_type(row[key])

            except Exception:
                print(f"[WARNING] Could not convert {key} -> keep default {vars(args)[key]}")

    return args


# ==========================
# main parser
# ==========================
def parameter_parser():
    parser = argparse.ArgumentParser(description="Run StrucGCN (config-driven)")

    # ALWAYS READ TXT (like original code)
    parser.add_argument("--config_path", type=str, default="new test.txt", help="Path to TSV config file")
    parser.add_argument("--id", type=str, default=None, help="Experiment id (optional). If not given, first row is used.")

    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--EPOCHS", type=int, default=1000)

    parser.add_argument("--train_ra", type=float, default=60)
    parser.add_argument("--val_ra", type=float, default=20)

    parser.add_argument('--k', type=int, default=16, help='Dimension of U and V matrices')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon parameter for U/V learning')
    parser.add_argument('--lamda', type=float, default=0.1, help='Lambda for structural features')
    parser.add_argument('--weight1', type=float, default=0.3, help='Weight for neighbor distribution')

    parser.add_argument("--dataset", default="actor")

    parser.add_argument("--feat_flag", type=str2bool, default=True)


    parser.add_argument("--backbone", type=str, default="gcn", help="Select model: gcn")


    parser.add_argument("--normalize", type=str, default="degree")
    parser.add_argument("--aggr", type=str, default="mean_concat")

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gama", type=float, default=0.3)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dr", type=float, default=0.4)
    parser.add_argument("--h_dim", type=int, default=64)

    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--role_generate", default="StrucWalkre")
    parser.add_argument("--feat_generate", default="onehot")

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--run_time", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=400)
    parser.add_argument("--force_regen", type=bool, default=False, help="Force regeneration of role matrices (Ignore cache)")
    parser.add_argument("--adaptive", type=str2bool, default=False, help="If True, alpha/beta/gama are learned via a trainable gating module.")

    args = parser.parse_args()

    runs = para_load(args.config_path)

    if len(runs) == 0:
        print("[WARNING] No config rows loaded -> using default args")
        print_args_as_table(args, title="ARGS (DEFAULT)")
        return args

    selected_row = None

    if args.id is None:
        selected_row = runs[0]
        args.id = selected_row.get("id", "unknown")
        print(f"[INFO] No --id given -> using FIRST row in config: {args.id}")

    else:
        for row in runs:
            if str(row.get("id")) == str(args.id):
                selected_row = row
                break

        if selected_row is None:
            print(f"[WARNING] id={args.id} not found -> using FIRST row instead")
            selected_row = runs[0]
            args.id = selected_row.get("id", "unknown")

    args = para_update(args, selected_row)

    print("\n==================== CONFIG APPLIED ====================")
    print("[INFO] config_path:", args.config_path)
    print("[INFO] selected id :", args.id)
    print("========================================================\n")

    print_args_as_table(args, title="ARGS (Final Used)")
    return args