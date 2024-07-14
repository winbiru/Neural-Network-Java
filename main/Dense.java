package main;

import java.util.Random;

import static main.Activation.*;

public class Dense extends Layer {
    private double[] input;
    private double[] z;
    private Activation activation;
    private int inputSize;
    private int outputSize;
    private Random random;

//    public Dense(int inputSize, int outputSize, String activation) {
//        super(inputSize, outputSize, activation);
//    }
    public Dense(int inputSize, int outputSize, Activation activation) {
        super(inputSize, outputSize, activation);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.random = new Random();
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        initializeWeights();
    }
    private void initializeWeights() {
        double range = Math.sqrt(6.0 / (inputSize + outputSize));
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextDouble() * 2 * range - range;
            }
        }
        for (int i = 0; i < outputSize; i++) {
            biases[i] = random.nextDouble() * 2 * range - range;
        }
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
