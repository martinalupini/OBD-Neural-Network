import matplotlib.pyplot as plt

def plotError(error_list, num_iterations, hidden_layers_activation_fn, with_momentum=True):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, error_list, marker='o', linestyle='-', color='b', linewidth=0.5)
    plt.xlabel('Number of training iterations')
    plt.ylabel('Error')
    plt.title('Error through training')
    plt.grid(True)
    #plt.show()
    if(with_momentum):
        plt.savefig('plots/error/' + hidden_layers_activation_fn + '_momentum_error.png')
    else:
        plt.savefig('plots/error/' + hidden_layers_activation_fn + '_error.png')
    plt.close()