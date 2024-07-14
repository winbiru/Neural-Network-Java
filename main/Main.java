package main;
import java.io.IOException;

import static main.Activation.*;

public class Main {
    public static void main(String[] args) throws IOException {
        // Load MNIST dataset
        double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
        int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");

        double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);

        NeuralNetwork model = new NeuralNetwork();
        model.addLayer(new Dense(784, 64, relu));
        model.addLayer(new Dense(64, 32, relu));
        model.addLayer(new Dense(32, 10, softmax));
        model.compile("categorical_crossentropy");

        // Huấn luyện mô hình
        model.fit(x_train, y_train, 10, 0.01);

        // Lưu trọng số
        model.saveWeights("model_weights.txt");

        // Tải trọng số
        model.loadWeights("model_weights.txt");
    }
}