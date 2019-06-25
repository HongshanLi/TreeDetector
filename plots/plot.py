import pickle
import matplotlib.pyplot as plt
from torchviz import make_dot



def train_val_acc(logfile):
    with open(logfile, 'rb') as f:
        log=pickle.load(f)
    
    train_acc = []
    val_acc = []
    for i in range(len(log)):
        train_acc.append(log[i+1]['train_acc'])
        val_acc.append(log[i+1]['val_acc'])

    num_epochs = len(log)
    
    train_acc, = plt.plot(range(num_epochs), train_acc, "b")
    val_acc, = plt.plot(range(num_epochs), val_acc, "r")


    plt.legend([train_acc, val_acc], ["Train Accuracy", "Val Accuracy"])

    plt.savefig("./train_val_acc.png")


if __name__=='__main__':
    train_val_acc("../unet_logs/log.pickle")

