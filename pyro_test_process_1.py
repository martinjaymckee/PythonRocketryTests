import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    filename = 'D:/Workspace/Rockets/PythonRocketryTests/data/pyro_test_dump.dat'

    df = pd.read_csv(filename)
    print(df)

    df.iloc[:, 5].plot()

    print('Ra = {} [+/- {}]'.format(df.iloc[:, 3].mean(), df.iloc[:, 3].std()))
    print('Rb = {} [+/- {}]'.format(df.iloc[:, 4].mean(), df.iloc[:, 4].std()))

    plt.show()