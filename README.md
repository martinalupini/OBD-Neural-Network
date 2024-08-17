# Implementation of a Neural Network from scratch

The project consists in the development of a neural network from scratch without relying on high-level libraries such as Keras or TensorFlow.    
The network is designed to be versatile and includes several advanced features:
1. **Regularization**: The network supports both L1 and L2 regularization methods. These techniques help prevent overfitting by penalizing large weights, ensuring the model generalizes well to unseen data.


2. **Cross-Validation**: It is used to optimize key hyperparameters of the network. This includes selecting the optimal number of hidden layers, determining the appropriate number of neurons per hidden layer, and finding the best regularization parameter. Cross-validation ensures that the chosen configuration achieves the best possible performance on the validation set before finalizing the model.


3. **Gradient Descent with Momentum**: For training, the network employs gradient descent with momentum. This optimization technique accelerates convergence and helps the network escape local minima, leading to faster and more reliable training.


4. **Dataset Flexibility**: The network is designed to handle different types of tasks, including binary classification and regression. This flexibility allows the model to be applied to a wide range of problems, from predicting binary outcomes to estimating continuous values.

---
Final project for the course **Optimization Methods for Big Data**, University of Rome Tor Vergata (Master's Degree in  Computer Engineering), September 2024.