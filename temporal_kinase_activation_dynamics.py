import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    time_points = ['6h', '24h', '48h', '72h']
    ERK = [2.1, 1.4, 2.7, 4.2]
    AKT = [1.0, 1.8, 3.0, 4.1]
    mTOR = [0.9, 1.7, 2.8, 3.9]
    plt.figure(figsize=(8,6))
    plt.plot(
    time_points,
    ERK,
    marker='o',
    linewidth=3,
    label='ERK'
    )
    plt.plot(
    time_points,
    AKT,
    marker='o',
    linewidth=3,
    label='AKT'
    )
    plt.plot(
    time_points,
    mTOR,
    marker='o',
    linewidth=3,
    label='mTOR'
    )
    plt.xlabel('Treatment Time')
    plt.ylabel('Kinase Activation Score')
    plt.title(
    'Temporal Kinase Activation Dynamics'
    )
    plt.legend()
    plt.show()
