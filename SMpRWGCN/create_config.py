import os

# =====================================================================
# راهنمای ترکیب‌های آلفا، بتا، گاما برای ۳ حالت معماری:
# حالت 110: alpha=0.5, beta=0.5, gama=0   (ترکیب خود گره و همسایگی محلی)
# حالت 101: alpha=0.5, beta=0, gama=0.5   (ترکیب خود گره و همسایگی ساختاری)
# حالت 011: alpha=0, beta=0.5, gama=0.5   (ترکیب همسایگی محلی و ساختاری)
# example

# id	N	train_ra	val_ra	dataset	feat_flag	backbone	normalize	aggr	alpha	beta	gama	lr	dr	h_dim	weight_decay	role_generate	feat_generate	seed	topk	force_regen
# gcn_tx_011_t	10	60	20	texas	True	gcn	degree	mean_concat	0	0.001	0.999	0.02	0.55	128	0.005	StrucWalkre	onehot	8	14	True
# gcn_cr_011_t	10	60	20	cornell	True	gcn	degree	mean_concat	0	0.001	0.999	0.02	0.55	128	0.005	StrucWalkre	onehot	8	14	True
# =====================================================================
base_configs = [
    # ==============================================================
    # 1. ACTOR (lr=0.02, dr=0.556, h_dim=128, wd=0.0001, topk=20, feat=node2vec, seed=8)
    # ==============================================================
    # --- با ویژگی (True) ---
    ("act_110_t", 10, 60, 20, "actor", "True", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),
    # ("act_101_t", 10, 60, 20, "actor", "True", "degree", "mean_concat", 0.2, 0.0, 0.8, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),
    # ("act_011_t", 10, 60, 20, "actor", "True", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),
    # --- بدون ویژگی (False) ---
    # ("act_110_f", 10, 60, 20, "actor", "False", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),
    # ("act_101_f", 10, 60, 20, "actor", "False", "degree", "mean_concat", 0.2, 0.0, 0.8, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),
    ("act_011_f", 10, 60, 20, "actor", "False", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.02, 0.556, 128, 0.0001, "StrucWalkre", "node2vec", 8, 20, "True"),

    # ==============================================================
    # 2. WISCONSIN (lr=0.01, dr=0.6, h_dim=128, wd=0.01, topk=18, seed=8)
    # ==============================================================
    ("wis_110_t", 10, 60, 20, "wisconsin", "True", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("wis_101_t", 10, 60, 20, "wisconsin", "True", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("wis_011_t", 10, 60, 20, "wisconsin", "True", "degree", "mean_concat", 0.0, 0.2, 0.8, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),
    ("wis_110_f", 10, 60, 20, "wisconsin", "False", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("wis_101_f", 10, 60, 20, "wisconsin", "False", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("wis_011_f", 10, 60, 20, "wisconsin", "False", "degree", "mean_concat", 0.0, 0.2, 0.8, 0.01, 0.6, 128, 0.01, "StrucWalkre", "onehot", 8, 18, "True"),

    # ==============================================================
    # 3. TEXAS (lr=0.005, dr=0.525, h_dim=128, wd=0.005, topk=20, seed=8)
    # ==============================================================
    ("tx_110_t", 10, 60, 20, "texas", "True", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),
    # ("tx_101_t", 10, 60, 20, "texas", "True", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),
    # ("tx_011_t", 10, 60, 20, "texas", "True", "degree", "mean_concat", 0.0, 0.9, 0.1, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),
    ("tx_110_f", 10, 60, 20, "texas", "False", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),
    # ("tx_101_f", 10, 60, 20, "texas", "False", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),
    # ("tx_011_f", 10, 60, 20, "texas", "False", "degree", "mean_concat", 0.0, 0.9, 0.1, 0.005, 0.525, 128, 0.005, "StrucWalkre", "onehot", 8, 20, "True"),

    # ==============================================================
    # 4. CORNELL (T: dr=0.55, h=64 | F: dr=0.2, h=128) - lr=0.01, wd=0.01, topk=18, seed=42
    # ==============================================================
    ("cr_110_t", 10, 60, 20, "cornell", "True", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.01, 0.55, 64, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),
    # ("cr_101_t", 10, 60, 20, "cornell", "True", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.01, 0.55, 64, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),
    # ("cr_011_t", 10, 60, 20, "cornell", "True", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.01, 0.55, 64, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),
    ("cr_110_f", 10, 60, 20, "cornell", "False", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.01, 0.2, 128, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),
    # ("cr_101_f", 10, 60, 20, "cornell", "False", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.01, 0.2, 128, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),
    # ("cr_011_f", 10, 60, 20, "cornell", "False", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.01, 0.2, 128, 0.01, "StrucWalkre", "onehot", 42, 18, "True"),

    # ==============================================================
    # 7. PUBMED
    # ==============================================================
    ("pubmed_110_t", 10, 60, 20, "pubmed", "True", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("pubmed_101_t", 10, 60, 20, "pubmed", "True", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "onehot", 8, 18, "True"),
    # ("pubmed_011_t", 10, 60, 20, "pubmed", "True", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "onehot", 8, 18, "True"),
    ("pubmed_110_f", 10, 60, 20, "pubmed", "False", "degree", "mean_concat", 0.5, 0.5, 0.0, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "node2vec", 8, 18, "True"),
    # ("pubmed_101_f", 10, 60, 20, "pubmed", "False", "degree", "mean_concat", 0.5, 0.0, 0.5, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "node2vec", 8, 18, "True"),
    # ("pubmed_011_f", 10, 60, 20, "pubmed", "False", "degree", "mean_concat", 0.0, 0.5, 0.5, 0.03, 0.556, 64, 0.0001, "StrucWalkre", "node2vec", 8, 18, "True"),
    ("cite_110_t", 10, 60, 20, "citeseer", "True", "degree", "mean_concat", 0.5, 0.5, 0, 0.01, 0.4, 64, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),
    # ("cite_101_t", 10, 60, 20, "citeseer", "True", "degree", "mean_concat", 0.5, 0, 0.5, 0.01, 0.4, 64, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),
    # ("cite_011_t", 10, 60, 20, "citeseer", "True", "degree", "mean_concat", 0.0, 0.99, 0.01, 0.01, 0.4, 64, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),
    # ("cite_110_f", 10, 60, 20, "citeseer", "False", "degree", "mean_concat", 0.5, 0.5, 0, 0.005, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),
    ("cite_101_f", 10, 60, 20, "citeseer", "False", "degree", "mean_concat", 0.5, 0, 0.5, 0.005, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),
    # ("cite_011_f", 10, 60, 20, "citeseer", "False", "degree", "mean_concat", 0.0, 0.999, 0.001, 0.005, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 42, 20, "True"),

    ("cham_110_t", 10, 60, 20, "chameleon", "True", "degree", "mean_concat", 0.5, 0.5, 0, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cham_101_t", 10, 60, 20, "chameleon", "True", "degree", "mean_concat", 0.02, 0.0, 0.82, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cham_011_t", 10, 60, 20, "chameleon", "True", "degree", "mean_concat", 0, 0.5, 0.5, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cham_110_f", 10, 60, 20, "chameleon", "False", "degree", "mean_concat", 0.5, 0.5, 0, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),
    ("cham_101_f", 10, 60, 20, "chameleon", "False", "degree", "mean_concat", 0.02, 0.0, 0.82, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cham_011_f", 10, 60, 20, "chameleon", "False", "degree", "mean_concat", 0, 0.5, 0.5, 0.015, 0.45, 128, 1e-05, "StrucWalkre", "onehot", 8, 16, "True"),

    ("cora_110_t", 10, 60, 20, "cora", "True", "degree", "mean_concat", 0.5, 0.5, 0, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cora_101_t", 10, 60, 20, "cora", "True", "degree", "mean_concat", 0.5, 0, 0.5, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cora_011_t", 10, 60, 20, "cora", "True", "degree", "mean_concat", 0, 0.5, 0.5, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cora_110_f", 10, 60, 20, "cora", "False", "degree", "mean_concat", 0.5, 0.5, 0, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),
    ("cora_101_f", 10, 60, 20, "cora", "False", "degree", "mean_concat", 0.5, 0, 0.5, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),
    # ("cora_011_f", 10, 60, 20, "cora", "False", "degree", "mean_concat", 0, 0.5, 0.5, 0.012, 0.55, 128, 0.0005, "StrucWalkre", "onehot", 8, 16, "True"),

    # ==============================================================
    # 8. SQUIRREL (بهترین تنظیمات پایه برای గرید سرچ: lr=0.03, dr=0.5, h_dim=128, topk=18)
    # ==============================================================
    # --- با ویژگی (True) ---
    ("sq_110_t", 10, 60, 20, "squirrel", "True", "degree", "mean_concat", 0.01, 0.99, 0.0, 0.03, 0.5, 128, 5e-06,
     "StrucWalkre", "onehot", 42, 18, "True"),
    # ("sq_101_t", 10, 60, 20, "squirrel", "True", "degree", "mean_concat", 0.01, 0.0, 0.99, 0.03, 0.5, 128, 5e-06,
    #  "StrucWalkre", "onehot", 42, 18, "True"),
    # ("sq_011_t", 10, 60, 20, "squirrel", "True", "degree", "mean_concat", 0.0, 0.01, 0.99, 0.03, 0.5, 128, 5e-06,
    #  "StrucWalkre", "onehot", 42, 18, "True"),

    # --- بدون ویژگی (False) ---
    ("sq_110_f", 10, 60, 20, "squirrel", "False", "degree", "mean_concat", 0.01, 0.99, 0.0, 0.03, 0.5, 128, 5e-06,
     "StrucWalkre", "onehot", 42, 18, "True"),
    # ("sq_101_f", 10, 60, 20, "squirrel", "False", "degree", "mean_concat", 0.01, 0.0, 0.99, 0.03, 0.5, 128, 5e-06,
    #  "StrucWalkre", "onehot", 42, 18, "True"),
    # ("sq_011_f", 10, 60, 20, "squirrel", "False", "degree", "mean_concat", 0.0, 0.01, 0.99, 0.03, 0.5, 128, 5e-06,
    #  "StrucWalkre", "onehot", 42, 18, "True"),
]
# فقط مدل gcn فعال است
models = ["gcn"]

# سربرگ فایل متنی
header = ["id", "N", "train_ra", "val_ra", "dataset", "feat_flag", "backbone",
          "normalize", "aggr", "alpha", "beta", "gama", "lr", "dr", "h_dim",
          "weight_decay", "role_generate", "feat_generate", "seed", "topk", "force_regen",]

output_lines = ["\t".join(header)]

# ساخت تنظیمات
for cfg in base_configs:
    base_id, n, train_ra, val_ra, dataset, feat_flag, norm, aggr, alpha, beta, gama, lr, dr, h_dim, wd, role_gen, feat_gen, seed, topk, force = cfg

    for model in models:
        new_id = f"{model}_{base_id}"


        row = [
            new_id, str(n), str(train_ra), str(val_ra), dataset, feat_flag, model,
            norm, aggr, str(alpha), str(beta), str(gama), str(lr), str(dr), str(h_dim),
            str(wd), role_gen, feat_gen, str(seed), str(topk), "True",
        ]

        output_lines.append("\t".join(row))

# ذخیره فایل
file_name = "new test.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines) + "\n")

print(f"✅ File '{file_name}' successfully created with Actor and 3 other Datasets!")