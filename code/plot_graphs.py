import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':

    save_dir = '../trained_models/graphs/'

    with open('saved_data_structures/epoch_losses_l_0.001_m_0.9.pickle', 'rb') as handle:
        epoch_losses = pickle.load(handle)

    with open('saved_data_structures/epoch_accuracies_l_0.001_m_0.9.pickle', 'rb') as handle:
        epoch_accuracies = pickle.load(handle)

    with open('saved_data_structures/val_acc_history_l_0.001_m_0.9.pickle', 'rb') as handle:
        val_acc_history = pickle.load(handle)

    print(epoch_losses)
    print(epoch_accuracies)
    print(val_acc_history)

    plt.title("Training loss vs number of epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.plot(np.arange(1,21), epoch_losses) #20 epochs
    plt.savefig(save_dir + 'Densenet_LossVsEpochs.png')
    plt.show()

    plt.title("Training accuracy vs number of epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.plot(np.arange(1,21), epoch_accuracies) #20 epochs
    plt.savefig(save_dir + 'Densenet_AccuracyVsEpochs.png')
    plt.show()

    plt.title("Validation accuracy vs number of epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Validation accuracy')
    plt.plot(np.arange(1,21), val_acc_history) #20 epochs
    plt.savefig(save_dir + 'Densenet_ValAccuracyVsEpochs.png')
    plt.show()
