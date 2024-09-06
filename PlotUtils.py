import matplotlib.pyplot as plt

def plotError(error_list, num_iterations, hidden_layers_activation_fn, with_momentum=True, model_name = ""):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, error_list, marker='', linestyle='-', color='b', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through training')
    plt.grid(True)
    #plt.show()
    if(model_name != ""):
        plt.savefig('plots/error/' + model_name + '.png')
        plt.close()
        return

    if(with_momentum):
        plt.savefig('plots/error/' + hidden_layers_activation_fn + '_momentum_error.png')
    else:
        plt.savefig('plots/error/' + hidden_layers_activation_fn + '_error.png')
    plt.close()


def plotAccuracy(accuracy_list, num_iterations, hidden_layers_activation_fn, with_momentum=True, model_name =""):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy_list, marker='', linestyle='-', color='r', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy through training')
    plt.grid(True)
    #plt.show()
    if(model_name != ""):
        plt.savefig('plots/accuracy/' + model_name + '.png')
        plt.close()
        return

    if(with_momentum):
        plt.savefig('plots/accuracy/' + hidden_layers_activation_fn + '_momentum_error.png')
    else:
        plt.savefig('plots/accuracy/' + hidden_layers_activation_fn + '_error.png')
    plt.close()