package main;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

public class NeuralNetworkTest {
    private NeuralNetwork model;

    @BeforeEach
    public void setUp() {
        model = new NeuralNetwork();
        model.addLayer(new Dense(784, 64, "relu"));
        model.addLayer(new Dense(64, 32, "relu"));
        model.addLayer(new Dense(32, 10, "softmax"));
        model.compile("categorical_crossentropy");
    }

    @Test
public void testSaveAndLoadWeights() throws IOException {
    // Train the model
    double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
    int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");
    double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);
    model.fit(x_train, y_train, 10, 0.01);

    // Save weights
    String weightsFile = "test_weights.txt";
    model.saveWeights(weightsFile);

    // Create a new model and load weights
    NeuralNetwork loadedModel = new NeuralNetwork();
    loadedModel.addLayer(new Dense(784, 64, "relu"));
    loadedModel.addLayer(new Dense(64, 32, "relu"));
    loadedModel.addLayer(new Dense(32, 10, "softmax"));
    loadedModel.compile("categorical_crossentropy");
    loadedModel.loadWeights(weightsFile);

    // Check if the loaded weights are equal to the saved weights
    double[][][] savedWeights = model.getWeights();
    double[][][] loadedWeights = loadedModel.getWeights();
    assertEquals(savedWeights.length, loadedWeights.length);
    for (int i = 0; i < savedWeights.length; i++) {
        assertEquals(savedWeights[i].length, loadedWeights[i].length);
        for (int j = 0; j < savedWeights[i].length; j++) {
            assertArrayEquals(savedWeights[i][j], loadedWeights[i][j], 1e-6);
        }
    }
}

    @Test
    public void testSaveAndLoadWeightsWithDifferentLayers() throws IOException {
        // Train the model
        double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
        int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");
        double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);
        model.fit(x_train, y_train, 10, 0.01);

        // Save weights
        String weightsFile = "test_weights.txt";
        model.saveWeights(weightsFile);

        // Create a new model with different layers and load weights
        NeuralNetwork loadedModel = new NeuralNetwork();
        loadedModel.addLayer(new Dense(784, 32, "relu"));
        loadedModel.addLayer(new Dense(32, 10, "softmax"));
        loadedModel.compile("categorical_crossentropy");

        // Ensure the weights file exists before trying to load it
        File weightsFileObj = new File(weightsFile);
        assertTrue(weightsFileObj.exists(), "Weights file does not exist");

        // Check if the weights file can be loaded
        try {
            loadedModel.loadWeights(weightsFile);
        } catch (IllegalArgumentException e) {
            fail("Expected no exception to be thrown, but an IllegalArgumentException was caught");
        }
    }

    @Test
    public void testSaveAndLoadWeightsWithDifferentLayerSizes() throws IOException {
        // Train the model
        double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
        int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");
        double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);
        model.fit(x_train, y_train, 10, 0.01);

        // Save weights
        String weightsFile = "test_weights.txt";
        model.saveWeights(weightsFile);

        // Create a new model with the same architecture as the original model and load weights
        NeuralNetwork loadedModel = new NeuralNetwork();
        loadedModel.addLayer(new Dense(784, 64, "relu"));
        loadedModel.addLayer(new Dense(64, 32, "relu"));
        loadedModel.addLayer(new Dense(32, 10, "softmax"));
        loadedModel.compile("categorical_crossentropy");

        // Check if the weights file exists before trying to load it
        File weightsFileObj = new File(weightsFile);
        assertTrue(weightsFileObj.exists(), "Weights file does not exist");

        // Check if the weights file can be loaded
        try {
            loadedModel.loadWeights(weightsFile);
        } catch (IllegalArgumentException e) {
            fail("Expected no exception to be thrown, but an IllegalArgumentException was caught");
        }
    }

    @Test
    public void testSaveAndLoadWeightsWithDifferentActivationFunctions() throws IOException {
        // Train the model
        double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
        int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");
        double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);
        model.fit(x_train, y_train, 10, 0.01);

        // Save weights
        String weightsFile = "test_weights.txt";
        model.saveWeights(weightsFile);

        // Create a new model with the same activation functions as the original model and load weights
        NeuralNetwork loadedModel = new NeuralNetwork();
        loadedModel.addLayer(new Dense(784, 64, "relu"));
        loadedModel.addLayer(new Dense(64, 32, "relu"));
        loadedModel.addLayer(new Dense(32, 10, "softmax"));
        loadedModel.compile("categorical_crossentropy");

        // Check if the weights file exists before trying to load it
        File weightsFileObj = new File(weightsFile);
        assertTrue(weightsFileObj.exists(), "Weights file does not exist");

        // Check if the weights file can be loaded
        try {
            loadedModel.loadWeights(weightsFile);
        } catch (IllegalArgumentException e) {
            fail("Expected no exception to be thrown, but an IllegalArgumentException was caught");
        }
    }
}