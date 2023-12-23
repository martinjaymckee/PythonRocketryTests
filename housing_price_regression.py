import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats


def load_data(filename):
    return pd.read_csv(filename)

def area_price_regression(data, N=10):
    xs_in = data['area']
    ys_in = data['price']
    xs = np.linspace(np.min(xs_in), np.max(xs_in), N)
    res = scipy.stats.linregress(xs_in, ys_in)
    ys = xs * res.slope + res.intercept
    return xs, ys, res.slope, res.intercept

def main():
    filename = 'Housing.csv'
    df = load_data(filename)
#    print(df)

    fig, ax = plt.subplots(1, constrained_layout=True)
    point_sizes = 7*df['bathrooms']
    sns.scatterplot(df, x='area', y='price', size=point_sizes, hue='bedrooms', ax=ax, alpha=0.7, legend=False, palette='viridis')
    xs, ys, _, _ = area_price_regression(df)
    sns.lineplot(x=xs, y=ys, ax=ax, c=(0.8, 0, 0))
    plt.show()

if __name__ == '__main__':
    main()
