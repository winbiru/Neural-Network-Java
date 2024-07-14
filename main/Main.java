package main;

import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

class Utils {
    // sum all items in array
    public static double sumArr(double[] arr) {
        double sum = 0;
        for (double val : arr) {
            sum += val;
        }
        return sum;
    }

    // random in range [min,max]
    public static double getRandomArbitrary(double min, double max) {
        return Math.random() * (max - min) + min;
    }

    public static double randn() {
        // Create random in range [0, 1)
        double u1 = Math.random();
        double u2 = Math.random();

        // Box-Muller Transform
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    public static double[][] transpose(double[][] weights) {
        int rows = weights.length;
        int cols = weights[0].length;
        double[][] result = new double[cols][rows];

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                result[i][j] = weights[j][i];
            }
        }

        return result;
    }

    public static double[][] heInit(int fanIn, int fanOut) {
        double scale = Math.sqrt(2.0 / fanIn);
        double[][] result = new double[fanIn][fanOut];

        for (int i = 0; i < fanIn; i++) {
            for (int j = 0; j < fanOut; j++) {
                result[i][j] = scale * randn();
            }
        }

        return result;
    }

    public static double[][] xavierInit(int fanIn, int fanOut) {
        double scale = Math.sqrt(6.0 / (fanIn + fanOut));
        double[][] result = new double[fanIn][fanOut];

        for (int i = 0; i < fanIn; i++) {
            for (int j = 0; j < fanOut; j++) {
                result[i][j] = scale * randn();
            }
        }

        return result;
    }

    public static double[] sigmoid(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = 1.0 / (1 + Math.exp(-arr[i]));
        }
        return result;
    }

    public static double[] sigmoidDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] * (1 - arr[i]);
        }
        return result;
    }

    public static double[] relu(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = Math.max(0, arr[i]);
        }
        return result;
    }

    public static double[] reluDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? 1 : 0;
        }
        return result;
    }

    public static double[] leakyRelu(double[] arr, double alpha) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? arr[i] : alpha * arr[i];
        }
        return result;
    }

    public static double[] leakyReluDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? 1 : 0.01;
        }
        return result;
    }

    public static double[] softmax(double[] arr) {
        double maxX = Arrays.stream(arr).max().getAsDouble();
        double[] expX = new double[arr.length];
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            expX[i] = Math.exp(arr[i] - maxX);
            sum += expX[i];
        }
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = expX[i] / sum;
        }
        return result;
    }

    public static double[] softmaxDerivative(double[] arr) {
        double[] s = softmax(arr);
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = s[i] * (1 - s[i]);
        }
        return result;
    }

    public static double mse(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += Math.pow(y_pred[i] - y_true[i], 2);
        }
        return sum / y_pred.length;
    }

    public static double[] mseGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = 2 * (y_pred[i] - y_true[i]) / y_pred.length;
        }
        return gradients;
    }

    public static double categoricalCrossEntropy(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += -y_true[i] * Math.log(y_pred[i] + 1e-15);
        }
        return sum;
    }

    public static double[] categoricalCrossEntropyGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = y_pred[i] - y_true[i];
        }
        return gradients;
    }

    public static double binaryCrossEntropy(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += y_true[i] * Math.log(y_pred[i] + 1e-15) + (1 - y_true[i]) * Math.log(1 - y_pred[i] + 1e-15);
        }
        return -sum / y_pred.length;
    }

    public static double[] binaryCrossEntropyGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = (y_pred[i] - y_true[i]) / ((y_pred[i] + 1e-15) * (1 - y_pred[i] + 1e-15));
        }
        return gradients;
    }

    public static void writeFile(String path, String data) throws IOException {
        Files.write(Paths.get(path), data.getBytes());
    }

    public static String readFile(String path) throws IOException {
        return new String(Files.readAllBytes(Paths.get(path)));
    }

    public static double[][] convertToOneHot(int[] labels, int numClasses) {
        double[][] oneHotVectors = new double[labels.length][numClasses];

        for (int i = 0; i < labels.length; i++) {
            oneHotVectors[i][labels[i]] = 1;
        }

        return oneHotVectors;
    }

    public static double[][] preprocessImages(byte[][] images) {
        double[][] result = new double[images.length][];
        for (int i = 0; i < images.length; i++) {
            result[i] = new double[images[i].length];
            for (int j = 0; j < images[i].length; j++) {
                result[i][j] = images[i][j] / 255.0;
            }
        }
        return result;
    }
}
abstract class Layer {
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
public class Main {
    public static void main(String[] args) throws IOException {
        // Load MNIST dataset
        double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
        int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");

        double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);

        NeuralNetwork model = new NeuralNetwork();
        model.addLayer(new Dense(784, 64, "relu"));
        model.addLayer(new Dense(64, 32, "relu"));
        model.addLayer(new Dense(32, 10, "softmax"));
        model.compile("categorical_crossentropy");

        // Huấn luyện mô hình
        model.fit(x_train, y_train, 10, 0.01);

        // Lưu trọng số
        model.saveWeights("model_weights.txt");

        // Tải trọng số
        model.loadWeights("model_weights.txt");
    }
}