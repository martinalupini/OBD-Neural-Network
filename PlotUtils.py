import matplotlib.pyplot as plt

def plotError(error_list, num_iterations, dir, model_name="final_model", activation_fn="relu"):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, error_list, marker='', linestyle='-', color='b', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through training')
    plt.grid(True)
    #plt.show()
    plt.savefig('plots/' + dir + '/' + activation_fn + '/error/' + model_name + '.png')
    plt.close()



def plotAccuracy(accuracy_list, num_iterations, dir, model_name="final_model", activation_fn="relu"):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy_list, marker='', linestyle='-', color='r', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy through training')
    plt.grid(True)
    #plt.show()
    plt.savefig('plots/' + dir + '/' + activation_fn + '/accuracy/' + model_name + '.png')
    plt.close()
