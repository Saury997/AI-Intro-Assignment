#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/28 下午8:46 
* Project: AI_Intro 
* File: exp5_ui.py
* IDE: PyCharm 
* Function: Application of Genetic Algorithm for Solving Travelling Salesman Problem (TSP) with UI Window
"""
import sys
import random
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QHBoxLayout, \
    QTextEdit, QVBoxLayout, QWidget, QPushButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 引入遗传算法和绘图模块
from algorithms import ga
from util import plot_ga_convergence, plot_route_ui

random.seed(42)

# 默认配置
config = {
    "pop_size": 500,  # 种群大小
    "n_gen": 200,  # 最大进化代数
    "sel_method": "tournament",  # 选择方法 tournament, roulette, uv
    "sel_size": 14,  # 锦标赛选择大小
    "mut_rate": 0.2,  # 变异率
    "mut_method": "reverse",  # 变异方法 swap, adj_swap, reverse, move
    "cov_method": "OX",  # 交叉方法 OX | err-> CX, PMX
    "cities_fn": "data/cities.xlsx",  # 城市数据文件路径
    "save_fig": False  # 保存图像
}


class GeneticAlgorithmThread(QThread):
    # 定义信号，用于向主线程发送数据
    finished = pyqtSignal(dict)  # 完成信号，发送一个字典（包含算法结果）

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        """执行遗传算法"""
        try:
            genes = ga.get_genes_from(self.config["cities_fn"])
            sys.stdout.write("-- Running TSP-GA with {} cities --\n".format(len(genes)))

            history = ga.run(
                genes,
                pop_size=self.config["pop_size"],
                max_generations=self.config["n_gen"],
                tourn_size=self.config["sel_size"],
                mut_rate=self.config["mut_rate"],
                mutation_type=self.config["mut_method"],
                sel_method=self.config["sel_method"],
                cov_method=self.config["cov_method"]
            )

            # 将结果通过信号返回给主线程
            self.finished.emit(history)
        except Exception as e:
            print(f"Error in GeneticAlgorithmThread: {e}")


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(text)

    def flush(self):
        pass


# GUI主窗口
class TSPGAWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genetic Algorithm - 解决旅行商问题TSP")
        self.setFixedSize(800, 850)  # 调整窗口大小以适应新的布局

        # 重定向标准输出
        sys.stdout = self.stream = Stream(newText=self.on_update_text)

        # 初始化界面组件
        layout = QVBoxLayout(self)

        # 参数设置区域和日志输出区域水平布局
        main_layout = QHBoxLayout()

        # 参数设置区域
        parameter_group = QWidget(self)
        parameter_layout = QVBoxLayout(parameter_group)
        parameter_group.setFixedSize(200, 350)  # 调整高度以适应新的布局

        # 种群大小
        self.pop_size_label = QLabel("种群大小:", self)
        self.pop_size_spin = QSpinBox(self)
        self.pop_size_spin.setValue(config["pop_size"])
        self.pop_size_spin.setRange(10, 1000)
        parameter_layout.addWidget(self.pop_size_label)
        parameter_layout.addWidget(self.pop_size_spin)

        # 最大进化代数
        self.n_gen_label = QLabel("最大进化代数:", self)
        self.n_gen_spin = QSpinBox(self)
        self.n_gen_spin.setValue(config["n_gen"])
        self.n_gen_spin.setRange(10, 1000)
        parameter_layout.addWidget(self.n_gen_label)
        parameter_layout.addWidget(self.n_gen_spin)

        # 选择方法
        self.sel_method_label = QLabel("选择方法:", self)
        self.sel_method_combobox = QComboBox(self)
        self.sel_method_combobox.addItems(["tournament", "roulette", "uv"])
        self.sel_method_combobox.setCurrentText(config["sel_method"])
        parameter_layout.addWidget(self.sel_method_label)
        parameter_layout.addWidget(self.sel_method_combobox)

        # 锦标赛选择大小
        self.sel_size_label = QLabel("锦标赛选择大小:", self)
        self.sel_size_spin = QSpinBox(self)
        self.sel_size_spin.setValue(config["sel_size"])
        self.sel_size_spin.setRange(2, 50)
        parameter_layout.addWidget(self.sel_size_label)
        parameter_layout.addWidget(self.sel_size_spin)

        # 变异率
        self.mut_rate_label = QLabel("变异率:", self)
        self.mut_rate_spin = QDoubleSpinBox(self)
        self.mut_rate_spin.setValue(config["mut_rate"])
        self.mut_rate_spin.setRange(0.0, 1.0)
        self.mut_rate_spin.setSingleStep(0.01)
        parameter_layout.addWidget(self.mut_rate_label)
        parameter_layout.addWidget(self.mut_rate_spin)

        # 变异方法
        self.mut_method_label = QLabel("变异方法:", self)
        self.mut_method_combobox = QComboBox(self)
        self.mut_method_combobox.addItems(["swap", "adj_swap", "reverse", "move"])
        self.mut_method_combobox.setCurrentText(config["mut_method"])
        parameter_layout.addWidget(self.mut_method_label)
        parameter_layout.addWidget(self.mut_method_combobox)

        author_label = QLabel("Designed by 宗介", self)
        author_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        author_label.setStyleSheet("font-size: 10pt; color: gray;")
        layout.addWidget(author_label, alignment=Qt.AlignBottom | Qt.AlignRight)

        # 交叉方法
        self.cov_method_label = QLabel("交叉方法:", self)
        self.cov_method_combobox = QComboBox(self)
        self.cov_method_combobox.addItems(["OX", "CX", "PMX"])
        self.cov_method_combobox.setCurrentText(config["cov_method"])
        parameter_layout.addWidget(self.cov_method_label)
        parameter_layout.addWidget(self.cov_method_combobox)

        # 将参数设置区域添加到主布局中
        main_layout.addWidget(parameter_group)

        # 日志输出区域
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)  # 设置为只读
        self.log_text.setFixedSize(500, 350)
        main_layout.addWidget(self.log_text)
        layout.addLayout(main_layout)

        # 执行按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.run_button = QPushButton("运行遗传算法", self)
        self.run_button.clicked.connect(self.run_ga)
        self.run_button.setFixedSize(150, 30)
        button_layout.addWidget(self.run_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        # 绘图区域
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def on_update_text(self, text):
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def run_ga(self):
        # 获取界面上设置的参数
        config["pop_size"] = self.pop_size_spin.value()
        config["n_gen"] = self.n_gen_spin.value()
        config["sel_method"] = self.sel_method_combobox.currentText()
        config["sel_size"] = self.sel_size_spin.value()
        config["mut_rate"] = self.mut_rate_spin.value()
        config["mut_method"] = self.mut_method_combobox.currentText()
        config["cov_method"] = self.cov_method_combobox.currentText()

        # 创建算法线程
        self.thread = GeneticAlgorithmThread(config=config)
        self.thread.finished.connect(self.on_ga_finished)  # 连接完成信号
        self.thread.start()  # 启动线程

    def on_ga_finished(self, history):
        """算法执行完成后更新UI"""
        try:
            # 更新UI，绘制路线图和收敛曲线
            self.draw_route(history['route'])
            self.draw_convergence(history['cost'])

            # 输出日志信息
            route = [gene.name for gene in history['route'].genes] + [history['route'].genes[0].name]
            self.log_text.append("遗传算法执行完成")
            self.log_text.append(f"最佳路径: {route}")
            self.log_text.append(f"最小代价: {min(history['cost'])}")
            self.log_text.append(f"-- Done --")
        except Exception as e:
            self.log_text.append(f"Error in on_ga_finished: {e}")

    def draw_route(self, route):
        """绘制路径图"""
        try:
            self.figure.clf()
            ax = self.figure.add_subplot(121)
            plot_route_ui(route, ax=ax)
            self.canvas.draw()
        except Exception as e:
            self.log_text.append(f"Error in draw_route: {e}")

    def draw_convergence(self, cost):
        """绘制收敛图"""
        try:
            ax = self.figure.add_subplot(122)
            ax.clear()
            plot_ga_convergence(cost, ax=ax, show=False)
            self.canvas.draw()
        except Exception as e:
            self.log_text.append(f"Error in draw_convergence: {e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TSPGAWindow()
    window.show()
    sys.exit(app.exec_())
