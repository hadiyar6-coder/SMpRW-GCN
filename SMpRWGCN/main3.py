import torch
import param_parser
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import utils_data
import os
import json
import utils
from model import SMpRWGNN
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def Train(args, trains, vals, neigh_data, struc_data, merged_data, path):
    net = SMpRWGNN(args, neigh_data, struc_data, merged_data).to(device)

    data = merged_data if neigh_data is None else neigh_data

    gcn_params = []
    fusion_params = []
    for name, param in net.named_parameters():
        if 'logit' in name:
            fusion_params.append(param)
        else:
            gcn_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': gcn_params, 'weight_decay': args.weight_decay},
        {'params': fusion_params, 'weight_decay': 0.0, 'lr': args.lr * 0.5}
    ], lr=args.lr)

    early_stopping_patience = getattr(args, "early_stopping_patience", 400)
    best_val_score = float('-inf')

    patience_counter = 0
    better_net = None
    epoch_times = []

    best_val_acc_for_table = 0.0

    for epoch in range(args.EPOCHS):
        epoch_start = time.time()

        net.train()

        logp = net()
        loss = F.nll_loss(logp[trains], data.y[trains])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            logp = net()
            acc = utils.accuracy(logp[vals], data.y[vals])
            val_loss = F.nll_loss(logp[vals], data.y[vals]).item()

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        score = acc
        is_better = score > best_val_score

        if is_better:
            best_val_score = score
            best_val_acc_for_table = acc
            patience_counter = 0
            better_net = {k: v.cpu() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print('Early stopping!')
                break

        if (epoch + 1) % 400 == 0 or (epoch + 1) == args.EPOCHS:
            print(
                f'Epoch [{epoch + 1}/{args.EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val acc: {acc:.4f}')

    net.load_state_dict({k: v.to(device) for k, v in better_net.items()})
    net.eval()

    with torch.no_grad():
        if hasattr(net, 'conv1') and hasattr(net, 'conv2'):
            def get_learned_coeffs(encoder_layer):
                if args.alpha == 0:
                    w = F.softmax(torch.stack([encoder_layer.beta_logit, encoder_layer.gama_logit], dim=0), dim=0)
                    return 0.0, w[0].mean().item(), w[1].mean().item()
                elif args.beta == 0:
                    w = F.softmax(torch.stack([encoder_layer.alpha_logit, encoder_layer.gama_logit], dim=0), dim=0)
                    return w[0].mean().item(), 0.0, w[1].mean().item()
                elif args.gama == 0:
                    w = F.softmax(torch.stack([encoder_layer.alpha_logit, encoder_layer.beta_logit], dim=0), dim=0)
                    return w[0].mean().item(), w[1].mean().item(), 0.0
                return args.alpha, args.beta, args.gama

            a1, b1, g1 = get_learned_coeffs(net.conv1)
            a2, b2, g2 = get_learned_coeffs(net.conv2)

            print(f'\n Learned Adaptive Coefficients (Best Epoch):')
            print(f'   - Initial (Config): Alpha={args.alpha:.4f}, Beta={args.beta:.4f}, Gamma={args.gama:.4f}')
            print(f'   - Layer 1 Learned : Alpha={a1:.4f}, Beta={b1:.4f}, Gamma={g1:.4f}')
            print(f'   - Layer 2 Learned : Alpha={a2:.4f}, Beta={b2:.4f}, Gamma={g2:.4f}\n')

            with open(path, 'a', encoding='utf-8') as file:
                file.write(f'Learned Adaptive Coefficients:\n')
                file.write(f'   - Layer 1: Alpha={a1:.4f}, Beta={b1:.4f}, Gamma={g1:.4f}\n')
                file.write(f'   - Layer 2: Alpha={a2:.4f}, Beta={b2:.4f}, Gamma={g2:.4f}\n\n')

    total_time = sum(epoch_times)
    mean_time = np.mean(epoch_times) if epoch_times else 0
    std_time = np.std(epoch_times) if epoch_times else 0

    print(f'⏱️ Mean Epoch Time: {mean_time:.4f} seconds ({mean_time * 1000:.2f} ms)')
    print(f'⏱️ Total Train Time: {total_time:.2f} seconds')
    print(f'⏱️ Std Epoch Time: {std_time:.4f} seconds')

    return better_net, net, mean_time, total_time, epoch_times, best_val_acc_for_table


def evaluate_node_classification(better_net, net, tests, labels):
    net.load_state_dict({k: v.to(device) for k, v in better_net.items()})
    net.eval()
    with torch.no_grad():
        logp = net()
        acc = utils.accuracy(logp[tests], labels[tests])
    return acc


if __name__ == "__main__":
    args = param_parser.parameter_parser()
    args_runs = param_parser.para_load(args.config_path)

    ACC_end_print = {}
    ACC_std_end_print = {}
    ACC_max_end_print = {}
    VAL_ACC_end_print = {}
    TIME_epoch_mean = {}
    TIME_total = {}
    ALL_TEST_RUNS = {}
    ALL_VAL_RUNS = {}

    for k in range(len(args_runs)):
        args = param_parser.para_update(args, args_runs[k])
        utils.set_seed(args.seed)

        backbone_name = getattr(args, 'backbone', 'gcn').upper()

        os.makedirs("C:/Users/hy/PycharmProjects/pythonProject6/bk3/SMpRW-GCN/result", exist_ok=True)
        path = os.path.join(
            'C:/Users/hy/PycharmProjects/pythonProject6/bk3/SMpRW-GCN/result/' +
            args.dataset + "_" + str(args_runs[k]['id']) + '.txt'
        )

        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(vars(args), indent=4))
            file.write("\n\n")

        print("\n===============================")
        print("RUN CONFIG:", args_runs[k]['id'])
        print("DATASET:", args.dataset)
        print("ACTIVE BACKBONE MODEL:", backbone_name)
        print("===============================\n")

        prep_start = time.time()

        node_map, graph, feat_data, labels = utils_data.prepare_data(args)

        if hasattr(args, "force_regen") and args.force_regen:
            role_adj_list_all, weigh_list_all = utils_data.generate_structure_similar_matrix(args, graph,
                                                                                             flag="generate")
        else:
            role_adj_list_all, weigh_list_all = utils_data.generate_structure_similar_matrix(args, graph, flag="load")

        prep_end = time.time()
        prep_time = prep_end - prep_start
        print(f'️ Data Preparation Time: {prep_time:.2f} seconds')

        if len(role_adj_list_all) == 0:
            print("[ERROR] role_adj_list_all is EMPTY -> check generate_structure_similar_matrix keys/save/load.")
            continue

        for method in sorted(role_adj_list_all.keys()):
            print("\n=====================================================")
            print(f"[INFO] Structural Method = {method}")
            print("=====================================================\n")

            with open(path, 'a', encoding='utf-8') as file:
                file.write("\n=====================================================\n")
                file.write(f"Structural Method = {method}\n")
                file.write(f"ACTIVE BACKBONE MODEL: {backbone_name}\n")
                file.write("=====================================================\n\n")

            role_adj_list = role_adj_list_all[method]
            weigh_list = weigh_list_all[method]

            Acc = []
            Val_Acc = []
            method_epoch_times = []
            method_total_time = 0

            method_start = time.time()

            # ساخت دیتای گراف یک بار در بیرون حلقه و اشتراک‌گذاری تنسور ویژگی x برای جلوگیری از OOM
            neigh_data, struc_data = utils_data.get_data(graph, feat_data, labels, role_adj_list, weigh_list)
            merged_data = utils_data.get_merged_data(args, neigh_data, struc_data)

            struc_data.x = None
            merged_data.x = None

            neigh_data = neigh_data.to(device)
            struc_data = struc_data.to(device)
            merged_data = merged_data.to(device)

            struc_data.x = neigh_data.x
            merged_data.x = neigh_data.x

            for i in range(args.N):
                with open(path, 'a', encoding='utf-8') as file:
                    file.write(f'\nexperiment: {i + 1}\n')
                print(f'\nexperiment: {i + 1}')

                exp_start = time.time()

                trains, vals, tests = utils_data.prepare_dataset(args, labels, i)

                trains = torch.from_numpy(trains).to(device)
                vals = torch.from_numpy(vals).to(device)
                tests = torch.from_numpy(tests).to(device)

                train_start = time.time()
                better_net, net, mean_epoch_time, total_train_time, epoch_times, val_acc_fold = Train(
                    args, trains, vals, neigh_data, struc_data, merged_data, path
                )
                train_end = time.time()

                method_epoch_times.extend(epoch_times)

                acc = evaluate_node_classification(better_net, net, tests, neigh_data.y)
                Acc.append(float(acc))
                Val_Acc.append(float(val_acc_fold))

                del net, better_net, trains, vals, tests
                torch.cuda.empty_cache()

                exp_end = time.time()
                exp_time = exp_end - exp_start

                print(f'️ Experiment {i + 1} Time: {exp_time:.2f} seconds')

            del neigh_data, struc_data, merged_data
            torch.cuda.empty_cache()

            method_end = time.time()
            method_total_time = method_end - method_start

            method_mean_epoch = np.mean(method_epoch_times) if method_epoch_times else 0
            method_std_epoch = np.std(method_epoch_times) if method_epoch_times else 0

            max_acc = float(np.max(Acc))
            mean_acc = float(np.mean(Acc))
            std_acc = float(np.std(Acc))
            mean_val_acc = float(np.mean(Val_Acc))

            key_name = f"Data:{args.dataset} | Model:{backbone_name} | ID:{args_runs[k]['id']}"

            ACC_max_end_print[key_name] = max_acc
            ACC_end_print[key_name] = mean_acc
            ACC_std_end_print[key_name] = std_acc
            VAL_ACC_end_print[key_name] = mean_val_acc
            TIME_epoch_mean[key_name] = method_mean_epoch * 1000
            TIME_total[key_name] = method_total_time

            ALL_TEST_RUNS[key_name] = Acc
            ALL_VAL_RUNS[key_name] = Val_Acc

            Acc_dictionary = {i: v for i, v in enumerate(Acc)}
            Val_dictionary = {i: v for i, v in enumerate(Val_Acc)}

            print('\n' + '=' * 50)
            print(f'FINAL RESULTS FOR: {method} ({backbone_name})')
            print('=' * 50)
            print(f' max Acc: {max_acc:.4f}')
            print(f' mean Acc (TEST): {mean_acc:.4f}')
            print(f' mean Acc (VALIDATION): {mean_val_acc:.4f}')
            print(f' std Acc: {std_acc:.4f}')
            print(f'️ Method Total Time: {method_total_time:.2f} seconds')
            print('-' * 50)
            print(f'️ 10 TEST RUNS: {Acc}')
            print(f'️ 10 VAL  RUNS: {Val_Acc}')
            print('=' * 50 + '\n')

            with open(path, 'a', encoding='utf-8') as file:
                file.write('max Acc: {:.4f}\n'.format(max_acc))
                file.write('mean Acc (TEST): {:.4f}\n'.format(mean_acc))
                file.write('mean Validation Acc: {:.4f}\n'.format(mean_val_acc))
                file.write('std Acc: {:.4f}\n'.format(std_acc))
                file.write('10 TEST RUNS: ' + json.dumps(Acc_dictionary) + '\n')
                file.write('10 VAL RUNS: ' + json.dumps(Val_dictionary) + '\n\n')

    # =========================================================================
    # چاپ جدول جمع‌بندی با سیستم ستاره‌دهی (ثبات انتخاب بر اساس Validation)
    # =========================================================================

    best_val_per_group = {}
    for key, val_acc in VAL_ACC_end_print.items():
        parts = key.split(" | ")
        dataset = parts[0].split(":")[1]
        model_type = parts[1].split(":")[1]
        group_key = f"{dataset}_{model_type}"

        if group_key not in best_val_per_group or val_acc > best_val_per_group[group_key]:
            best_val_per_group[group_key] = val_acc

    summary_text = "\n" + "=" * 120 + "\n"
    summary_text += " FINAL EXPERIMENT RESULTS SUMMARY (★ = Best Validation Selected Model) \n"
    summary_text += "=" * 120 + "\n"
    summary_text += f"{'Dataset':<12} | {'Variant / ID':<25} | {'Backbone':<10} | {'Val Acc':<11} | {'Test Mean':<10} | {'Test Max':<10} | {'Std':<8} | {'Total Time'}\n"
    summary_text += "-" * 120 + "\n"

    for key in ACC_end_print.keys():
        parts = key.split(" | ")
        dataset = parts[0].split(":")[1]
        model_type = parts[1].split(":")[1]
        variant_id = parts[2].split(":")[1]
        group_key = f"{dataset}_{model_type}"

        val_mean = VAL_ACC_end_print[key]
        test_mean = ACC_end_print[key]
        test_max = ACC_max_end_print[key]
        std_v = ACC_std_end_print[key]
        tot_time = TIME_total[key]

        star = "★" if abs(val_mean - best_val_per_group[group_key]) < 1e-6 else " "

        summary_text += f"{dataset:<12} | {variant_id[:25]:<25} | {model_type:<10} | {val_mean * 100:>6.2f}% {star} | {test_mean * 100:>6.2f}%   | {test_max * 100:>6.2f}%  | {std_v * 100:>5.2f}% | {tot_time:<10.2f}s\n"

    summary_text += "=" * 120 + "\n"

    print(summary_text)

    path_end = os.path.join('end.txt')
    with open(path_end, 'w', encoding='utf-8') as file:
        file.write(summary_text)

        file.write("\n\n=== 10 TEST RUNS (For Statistical Analysis) ===\n")
        file.write(json.dumps(ALL_TEST_RUNS, indent=2))

        file.write("\n\n=== 10 VALIDATION RUNS ===\n")
        file.write(json.dumps(ALL_VAL_RUNS, indent=2))

        file.write("\n\n=== DETAILED DICTIONARIES ===\n")
        file.write("\n=== VALIDATION RESULTS ===\n")
        file.write(json.dumps(VAL_ACC_end_print, indent=2))
        file.write("\n=== ACCURACY RESULTS ===\n")
        file.write(json.dumps(ACC_end_print, indent=2))
        file.write('\n\n=== STD RESULTS ===\n')
        file.write(json.dumps(ACC_std_end_print, indent=2))
        file.write('\n\n=== TIME RESULTS (ms/epoch) ===\n')
        file.write(json.dumps(TIME_epoch_mean, indent=2))
        file.write('\n\n=== TOTAL TIME (seconds) ===\n')
        file.write(json.dumps(TIME_total, indent=2))

    print("\n[INFO] Summary table and ALL 10 RUNS saved successfully to 'end.txt'")