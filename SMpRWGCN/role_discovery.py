import time
from SMpRW import StrucWalkre


class Role:
    def __init__(self, args, graph):
        self.args = args
        self.graph = graph

        # deepwalk:
        # self.methods = ["deepwalk"]


        self.methods = ["node2vec"]

        # self.methods = ["walklets", "rwr", "non_backtracking"]

        # self.methods = ["deepwalk", "walklets", "rwr", "non_backtracking"]

    def get_role_adj_weigh(self):

        role_adj_list_all = {}
        weigh_list_all = {}

        t0 = time.time()

        for m in self.methods:
            print(f"\n[INFO] Running SMpRW with RW method: {m}")

            model = StrucWalkre(
                self.graph,
                workers=self.args.workers,
                topk=self.args.topk,
                rw_method=m
            )

            role_adj_list, weigh_list = model.get_similarity_weigh()

            role_adj_list_all[m] = role_adj_list
            weigh_list_all[m] = weigh_list

        print(f"\n[INFO] Total S1 generation time: {time.time() - t0:.2f}s")
        return role_adj_list_all, weigh_list_all
