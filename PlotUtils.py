import matplotlib.pyplot as plt
import numpy as np


def plotError(error_list, num_iterations, dir, model_name="final_model_error", activation_fn="relu"):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, error_list, marker='', linestyle='-', color='b', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through training')
    plt.grid(True)
    plt.savefig('plots/' + dir + '/' + activation_fn + "/" + model_name + '.png')
    #plt.show()
    plt.close()



def plotAccuracy(accuracy_list, num_iterations, dir, model_name="final_model_accuracy", activation_fn="relu"):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy_list, marker='', linestyle='-', color='r', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy through training')
    plt.grid(True)
    #plt.show()
    plt.savefig('plots/' + dir + '/' + activation_fn + "/" + model_name + '.png')
    plt.close()


def plotClasses(count, file_name):
    plt.bar(['Negative', 'Positive'], [count.get(0, 0), count.get(1, 0)], color=['red', 'green'])
    plt.xlabel('Class')
    plt.ylabel('Number of instances')
    plt.title('Distribution of positive and negative instances')
    plt.savefig('plots/' + file_name)
    plt.close()


def exploratory_data_analysis(y, file_name):

    plt.bar(['Negative', 'Positive'], [np.sum(y == 0), np.sum(y == 1)], color=['red', 'green'])
    plt.xlabel('Class')
    plt.ylabel('Number of instances')
    plt.title('Distribution of positive and negative instances')
    plt.savefig('plots/' + file_name)
    plt.close()
