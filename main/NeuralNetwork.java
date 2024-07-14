package main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

public class NeuralNetwork {
    List<Layer> layers;
    BiFunction<double[], double[], Double> lossFunc;
    BiFunction<double[], double[], double[]> lossDerivative;

    public NeuralNetwork() {
        this.layers = new ArrayList<>();
    }

    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public void compile(String lossFunc) {
        switch (lossFunc) {
            case "mse":
                this.lossFunc = Utils::mse;
                this.lossDerivative = Utils::mseGradient;
                break;
            case "categorical_crossentropy":
                this.lossFunc = Utils::categoricalCrossEntropy;
                this.lossDerivative = Utils::categoricalCrossEntropyGradient;
                break;
            case "binary_crossentropy":
                this.lossFunc = Utils::binaryCrossEntropy;
                this.lossDerivative = Utils::binaryCrossEntropyGradient;
                break;
        }
    }

    public double[] predict(double[] inputs) {
        double[] result = inputs;
        for (Layer layer : layers) {
            result = layer.forward(result);
        }
        return result;
    }

    public void fit(double[][] x_train, double[][] y_train, int epochs, double lr) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = 0;
            int correctPredictions = 0;
            int totalIterations = x_train.length;

            for (int i = 0; i < totalIterations; i++) {
                double[] output = predict(x_train[i]);
                loss += lossFunc.apply(output, y_train[i]);

                // Tính số dự đoán chính xác
                int predictedClass = getPredictedClass(output);
                int trueClass = getTrueClass(y_train[i]);
                if (predictedClass == trueClass) {
                    correctPredictions++;
                }

                double[] error = lossDerivative.apply(output, y_train[i]);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    error = layers.get(j).backward(error, lr);
                }

                // Hiển thị tiến độ
                double progress = (i + 1) / (double) totalIterations * 100;
                System.out.printf("\rEpoch: %d, Progress: %.2f%%", epoch + 1, progress);
            }

            loss /= totalIterations;
            double accuracy = (double) correctPredictions / totalIterations * 100;

            System.out.printf("\nEpoch: %d, Loss: %.10f, Accuracy: %.2f%%\n", epoch + 1, loss, accuracy);
        }
    }

    private int getPredictedClass(double[] output) {
        int maxIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private int getTrueClass(double[] oneHotVector) {
        for (int i = 0; i < oneHotVector.length; i++) {
            if (oneHotVector[i] == 1.0) {
                return i;
            }
        }
        return -1; // Không tìm thấy
    }

    void saveWeights(String filePath) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(Arrays.deepToString(layer.weights)).append("\n");
            sb.append(Arrays.toString(layer.biases)).append("\n");
        }
        Utils.writeFile(filePath, sb.toString());
    }

    void loadWeights(String filePath) throws IOException {
        String[] data = Utils.readFile(filePath).split("\n");
        int idx = 0;
        for (Layer layer : layers) {
            String[] weightsData = data[idx].replace("[[", "").replace("]]", "").split("], \\[");
            double[][] weights = new double[weightsData.length][weightsData[0].split(", ").length];
            for (int i = 0; i < weightsData.length; i++) {
                String[] weightRow = weightsData[i].split(", ");
                for (int j = 0; j < weightRow.length; j++) {
                    weights[i][j] = Double.parseDouble(weightRow[j]);
                }
            }
            layer.weights = weights;

            String[] biasesData = data[idx + 1].replace("[", "").replace("]", "").split(", ");
            double[] biases = new double[biasesData.length];
            for (int i = 0; i < biasesData.length; i++) {
                biases[i] = Double.parseDouble(biasesData[i]);
            }
            layer.biases = biases;

            idx += 2;
        }
    }
}
