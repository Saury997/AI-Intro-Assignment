#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/28 下午4:03 
* Project: AI_Intro 
* File: ga.py
* IDE: PyCharm 
* Function: Genetic Algorithm specially for TSP
"""
import sys
from time import time
import pandas as pd
from haversine import haversine
from tqdm import tqdm
import random


def get_genes_from(fn, sample_n=0):
    df = pd.read_excel(fn)
    genes = [Gene(row['city'], row['latitude'], row['longitude'])
             for _, row in df.iterrows()]

    return genes if sample_n <= 0 else random.sample(genes, sample_n)


class Gene:  # City
    __distances_table = {}

    def __init__(self, name, lat, lng):
        self.name = name
        self.lat = lat
        self.lng = lng

    def get_distance_to(self, dest):
        origin = (self.lat, self.lng)
        dest = (dest.lat, dest.lng)

        forward_key = origin + dest
        backward_key = dest + origin

        if forward_key in Gene.__distances_table:
            return Gene.__distances_table[forward_key]

        if backward_key in Gene.__distances_table:
            return Gene.__distances_table[backward_key]

        dist = int(haversine(origin, dest))
        Gene.__distances_table[forward_key] = dist

        return dist


class Individual:
    def __init__(self, genes):
        assert len(genes) > 3
        self.genes = genes
        self.__reset_params()

    def swap(self, gene_1, gene_2):
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def reverse_segment(self, idx1, idx2):
        """
        反转基因序列的一个区间
        :param idx1: 起始索引
        :param idx2: 结束索引
        """
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1  # 确保 idx1 < idx2
        self.genes[idx1:idx2 + 1] = self.genes[idx1:idx2 + 1][::-1]  # 反转区间
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            self.__fitness = 1 / (self.travel_cost + 0.0001)  # Normalize travel cost
        return self.__fitness

    @property
    def travel_cost(self):  # Get total travelling cost
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i + 1]

                self.__travel_cost += origin.get_distance_to(dest)

        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0


class Population:  # Population of individuals
    def __init__(self, individuals):
        self.individuals = individuals

    @staticmethod
    def gen_individuals(sz, genes):
        individuals = []
        for _ in range(sz):
            individuals.append(Individual(random.sample(genes, len(genes))))
        return Population(individuals)

    def add(self, route):
        self.individuals.append(route)

    def rmv(self, route):
        self.individuals.remove(route)

    def get_fittest(self):
        fittest = self.individuals[0]
        for route in self.individuals:
            if route.fitness > fittest.fitness:
                fittest = route

        return fittest


def evolve(pop, tourn_size, mut_rate, mutation_type, sel_method, cov_method):
    new_generation = Population([])
    pop_size = len(pop.individuals)
    elitism_num = int(pop_size * 0.2)  # 取 20% 作为精英

    # 精英
    for _ in range(elitism_num):
        fittest = pop.get_fittest()
        new_generation.add(fittest)
        pop.rmv(fittest)

    # 交叉
    for _ in range(elitism_num, pop_size):
        parent_1 = selection(new_generation, tourn_size, sel_method)
        parent_2 = selection(new_generation, tourn_size, sel_method)
        child = crossover(parent_1, parent_2, cov_method)
        new_generation.add(child)

    # 变异
    for i in range(elitism_num, pop_size):
        mutate(new_generation.individuals[i], mut_rate, mutation_type)

    return new_generation


def crossover(parent_1, parent_2, method="OX"):
    """交叉操作，根据指定的方式生成子代"""

    def ox_crossover(parent_1, parent_2):
        """顺序交叉（OX）"""
        genes_n = len(parent_1.genes)
        child = Individual([None for _ in range(genes_n)])

        # Step 1: Randomly select a subsequence from parent 1 and copy it to the child
        start_at = random.randint(0, genes_n - genes_n // 2 - 1)
        finish_at = start_at + genes_n // 2
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

        # Step 2: Fill the remaining positions with the genes of parent 2, ensuring no duplicates
        j = 0
        for i in range(genes_n):
            if child.genes[i] is None:
                while parent_2.genes[j] in child.genes:
                    j += 1
                child.genes[i] = parent_2.genes[j]
                j += 1

        return child

    def pmx_crossover(parent_1, parent_2):
        """部分交叉（PMX）"""
        genes_n = len(parent_1.genes)
        child = Individual([None for _ in range(genes_n)])

        # Step 1: Select a random segment from parent_1
        start_at = random.randint(0, genes_n - genes_n // 2 - 1)
        finish_at = start_at + genes_n // 2
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

        # Step 2: Create a mapping between the genes in the crossover segment
        mapping = {parent_1.genes[i]: parent_2.genes[i] for i in range(start_at, finish_at)}

        # Step 3: Fill in the remaining positions from parent_2 and apply the mapping
        for i in range(genes_n):
            if child.genes[i] is None:
                gene_from_p2 = parent_2.genes[i]
                # If gene_from_p2 is in the mapping, use the mapped gene from parent_1
                if gene_from_p2 in mapping:
                    child.genes[i] = mapping[gene_from_p2]
                else:
                    child.genes[i] = gene_from_p2

        return child

    def cx_crossover(parent_1, parent_2):
        """循环交叉（CX）"""
        genes_n = len(parent_1.genes)
        child = Individual([None for _ in range(genes_n)])

        # Step 1: Start with the first gene from parent_1
        child.genes[0] = parent_1.genes[0]

        # Step 2: Use cycle to fill in child genes
        i = 0
        cycle_start = parent_1.genes[0]
        while None in child.genes:
            current_gene = parent_1.genes[i]
            # Find the corresponding gene in parent_2
            idx_in_parent2 = parent_2.genes.index(current_gene)
            if child.genes[idx_in_parent2] is None:  # Only fill if it's None
                child.genes[idx_in_parent2] = parent_2.genes[idx_in_parent2]

            # Move to the next gene in the cycle based on parent_1's gene at the current index
            i = idx_in_parent2
            if parent_1.genes[i] == cycle_start:
                break  # Once we return to the start of the cycle, break the loop

        return child

    # 根据选择的交叉方式，调用相应的交叉函数
    if method == "OX":
        return ox_crossover(parent_1, parent_2)
    elif method == "PMX":
        return pmx_crossover(parent_1, parent_2)
    elif method == "CX":
        return cx_crossover(parent_1, parent_2)
    else:
        raise ValueError(f"Unknown crossover method: {method}")


def mutate(individual, rate, mutation_type='swap'):
    """变异方法，根据mutation_type选择不同的变异操作"""
    for _ in range(len(individual.genes)):
        if random.random() < rate:
            if mutation_type == 'swap':
                # 两点交换：选择两个基因并交换它们的位置
                sel_genes = random.sample(individual.genes, 2)
                individual.swap(sel_genes[0], sel_genes[1])
            elif mutation_type == 'adj_swap':
                # 相邻互换：选择两个相邻的基因并交换它们的位置
                idx = random.randint(0, len(individual.genes) - 2)  # 随机选择一个位置
                individual.swap(individual.genes[idx], individual.genes[idx + 1])
            elif mutation_type == 'reverse':
                # 区间逆转：选择一个区间并逆转该区间的基因顺序
                idx1 = random.randint(0, len(individual.genes) - 1)
                idx2 = random.randint(idx1 + 1, len(individual.genes))  # 保证idx2 > idx1
                individual.reverse_segment(idx1, idx2)
            elif mutation_type == 'move':
                # 单点移动：选择一个基因，并将其移动到随机位置
                idx = random.randint(0, len(individual.genes) - 1)
                gene_to_move = individual.genes.pop(idx)  # 删除该基因
                new_pos = random.randint(0, len(individual.genes))  # 随机选择新位置
                individual.genes.insert(new_pos, gene_to_move)  # 插入基因到新位置
            else:
                raise ValueError(f"Unknown mutation type: {mutation_type}")


def selection(population, competitors_n=5, sel_method="tournament"):
    """根据选择方法选择个体"""

    if sel_method == "tournament":
        # 锦标赛选择
        return Population(random.sample(population.individuals, competitors_n)).get_fittest()

    elif sel_method == "roulette":
        # 轮盘赌选择（适应度比例选择）
        total_fitness = sum(individual.fitness for individual in population.individuals)
        pick = random.uniform(0, total_fitness)
        current = 0

        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                return individual

    elif sel_method == "uv":
        # 均匀选择
        return random.choice(population.individuals)

    else:
        raise ValueError(f"未知的选择方法: {sel_method}")


def run(genes, pop_size, max_generations, tourn_size, mut_rate, mutation_type, sel_method, cov_method):
    population = Population.gen_individuals(pop_size, genes)
    history = {'cost': [population.get_fittest().travel_cost]}
    counter, generations, min_cost = 0, 0, float('inf')

    start_time = time()
    with tqdm(total=max_generations, desc="GA Progress", ncols=120, file=sys.stdout) as pbar:
        while generations < max_generations:
            # 进化种群
            population = evolve(population, tourn_size, mut_rate, mutation_type, sel_method, cov_method)
            cost = population.get_fittest().travel_cost

            # 更新最优值及停滞计数器
            if cost < min_cost:
                counter, min_cost = 0, cost
            else:
                counter += 1

            # 更新历史记录
            generations += 1
            history['cost'].append(cost)

            # 更新进度条
            pbar.update(1)  # 始终基于 generations 更新进度条
            pbar.set_postfix({"min_cost": f"{min_cost:.2f}", "counter": counter})

            # # 提前停止条件
            # if counter >= max_generations // 3:  # 停滞超过总代数的 33% 时提前终止
            #     print("Early stopping due to stagnation.")
            #     break

    total_time = round(time() - start_time, 6)

    sys.stdout.write(f"Evolution finished after {generations} generations in {total_time} s\n")
    sys.stdout.write(f"Minimum travelling cost {min_cost} KM\n")

    history['generations'] = generations
    history['total_time'] = total_time
    history['route'] = population.get_fittest()

    return history

