import matplotlib.pyplot as plt

def acc_plot(train_acc, test_acc):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.show()