import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
df = pd.read_csv('/')  #Specify the directory
df.corr()

sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()



