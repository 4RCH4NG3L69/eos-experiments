# Code Documentation

### Implementation Guidelines

1. **Architectures**
    * Fully connected neural networks (FCNN)
    * Convolutional neural network (CNN)
      * Visual Geometry Group (VGG)
      * Residual neural network (ResNet) / 
    * Recurrent neural network (RNN) (maybe not)
    * Transformer
2. **Activation Functions**
   * ReLU
   * Sigmoid
   * tanh
   * Softma
   * Leaky ReLU
3. **Loss Functions**
   * Mean Squared Error (MSE)
   * Cross-Entropy (CE)
4. **Data Acquisition**
   * Length of trajectory, i.e. the length of the iterate sequence
   * "Principal" parameters, i.e. find out if there are any weights/biases which have a major effect on the trajectory
   * Higher order eigenvalues in Hessian, i.e. see what happens to other eigenvalues
   * Train loss (from paper)
   * Sharpness, i.e. maximum eigenvalue of Hessian (from paper)
5. **Optimizers**
   * Full-batch gradient descent (GD)
   * Stochastic gradient descent (SGD)
   * Adam
6. **Parameter Initializations**
   * Uniform initialization
    * Xavier initialization
    * Kaiming initialization
    * Zeros initialization
    * One’s initialization
    * Normal initialization