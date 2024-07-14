package main;

public abstract class Layer {
    double[][] weights;
    double[] biases;
    String activation;

    public Layer(int inputSize, int outputSize, String activation) {
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        this.activation = activation;

        // Initialize weights and biases
        initializeWeightsAndBiases(inputSize, outputSize);
    }

    public abstract double[] forward(double[] input);

    public abstract double[] backward(double[] dA, double learningRate);

    protected void initializeWeightsAndBiases(int inputSize, int outputSize) {
        this.weights = Utils.heInit(inputSize, outputSize);
        this.biases = new double[outputSize];
    }

    protected double[] applyActivation(double[] z) {
        switch (activation) {
            case "sigmoid":
                return Utils.sigmoid(z);
            case "relu":
                return Utils.relu(z);
            case "leaky_relu":
                return Utils.leakyRelu(z, 0.01);
            case "softmax":
                return Utils.softmax(z);
            default:
                return z;
        }
    }

    protected double[] applyActivationDerivative(double[] z) {
        switch (activation) {
            case "sigmoid":
                return Utils.sigmoidDerivative(z);
            case "relu":
                return Utils.reluDerivative(z);
            case "leaky_relu":
                return Utils.leakyReluDerivative(z);
            case "softmax":
                return Utils.softmaxDerivative(z);
            default:
                return z;
        }
    }
}
