#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/27 下午9:36 
* Project: AI_Intro 
* File: exp5.py
* IDE: PyCharm 
* Function: Application of Genetic Algorithm for Solving Travelling Salesman Problem (TSP)
"""
import random
from algorithms import ga
from util import plot_ga_convergence, plot_route

random.seed(42)

config = {
    "pop_size": 500,  # 种群大小
    "n_gen": 200,  # 最大进化代数
    "sel_method": "tournament",  # 选择方法 tournament, roulette, uv
    "sel_size": 14,  # 锦标赛选择大小
    "mut_rate": 0.2,  # 变异率
    "mut_method": "reverse",    # 变异方法 swap, adj_swap, reverse, move
    "cov_method": "OX",  # 交叉方法 OX | err-> CX, PMX
    "cities_fn": "data/cities.xlsx",  # 城市数据文件路径
    "save_fig": True  # 保存图像
}


if __name__ == "__main__":
    genes = ga.get_genes_from(config["cities_fn"])

    print("-- Running TSP-GA with {} cities --".format(len(genes)))

    history = ga.run(
        genes,
        pop_size=config["pop_size"],
        max_generations=config["n_gen"],
        tourn_size=config["sel_size"],
        mut_rate=config["mut_rate"],
        mutation_type=config["mut_method"],
        sel_method=config["sel_method"],
        cov_method=config["cov_method"]
    )

    print("-- Drawing Route --")

    plot_route(history['route'], save_fig=config["save_fig"])
    plot_ga_convergence(history['cost'], save_fig=config["save_fig"])

    print("-- Done --")
