import pickle
import matplotlib.pyplot as plt

def train_val_acc(logfile):
    with open(logfile, 'rb') as f:
        log=pickle.load(f)
    
    train_acc = []
    val_acc = []
    for i in range(len(log)):
        train_acc.append(log[i+1]['train_)
        val_acc.append(log[i+1]


if __name__=='__main__':
    train_val_loss("../logs/log.pickle")

