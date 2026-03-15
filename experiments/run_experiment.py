"""
智能计算系统课程实验运行脚本

使用方法：
    python experiments/run_experiment.py --lab <实验编号>
    
示例：
    python experiments/run_experiment.py --lab 01
"""

import argparse
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_lab(lab_number):
    """
    运行指定编号的实验
    
    Args:
        lab_number: 实验编号（字符串，如 '01'）
    """
    print(f"开始运行实验 {lab_number}...")
    
    # 在这里添加实验运行逻辑
    # 例如：
    # if lab_number == '01':
    #     from src.algorithms.lab01 import main
    #     main()
    # elif lab_number == '02':
    #     from src.algorithms.lab02 import main
    #     main()
    # else:
    #     print(f"实验 {lab_number} 尚未实现")
    
    print(f"实验 {lab_number} 运行完成！")


def main():
    parser = argparse.ArgumentParser(description='运行智能计算系统课程实验')
    parser.add_argument(
        '--lab',
        type=str,
        required=True,
        help='实验编号（如：01, 02）'
    )
    
    args = parser.parse_args()
    run_lab(args.lab)


if __name__ == '__main__':
    main()
