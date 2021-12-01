import matplotlib as plt
def plot(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*[X[:, i] for i in range(X.shape[1])], y)
    plt.show()
