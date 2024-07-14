package main;

public class Dense extends Layer {
    private double[] input;
    private double[] z;

    public Dense(int inputSize, int outputSize, String activation) {
        super(inputSize, outputSize, activation);
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;
        this.z = new double[biases.length];
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < input.length; j++) {
                z[i] += input[j] * weights[j][i];
            }
            z[i] += biases[i];
        }
        return applyActivation(z);
    }

    @Override
    public double[] backward(double[] dA, double learningRate) {
        double[] dZ = new double[dA.length];
        double[] dInput = new double[input.length];
        for (int i = 0; i < dA.length; i++) {
            dZ[i] = dA[i] * applyActivationDerivative(z)[i];
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < dZ.length; j++) {
                dInput[i] += dZ[j] * weights[i][j];
                weights[i][j] -= learningRate * dZ[j] * input[i];
            }
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * dZ[i];
        }
        return dInput;
    }
}
