package main;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Unittest {
        @Test
        public void testPredictionOnUnseenData() throws IOException {
            // Load MNIST dataset
            double[][] x_train = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
            int[] y_train_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");

            double[][] x_test = MNISTLoader.loadImages("src/mnist-data/train-images.idx3-ubyte");
            int[] y_test_labels = MNISTLoader.loadLabels("src/mnist-data/train-labels.idx1-ubyte");

            double[][] y_train = MNISTLoader.convertLabelsToOneHot(y_train_labels, 10);
            double[][] y_test = MNISTLoader.convertLabelsToOneHot(y_test_labels, 10);

            // Train the model
            NeuralNetwork model = new NeuralNetwork();
            model.addLayer(new Dense(784, 64, "relu"));
            model.addLayer(new Dense(64, 32, "relu"));
            model.addLayer(new Dense(32, 10, "softmax"));
            model.compile("categorical_crossentropy");
            model.fit(x_train, y_train, 10, 0.01);

            // Make predictions on unseen data
            for (int i = 0; i < 10; i++) {
                double[] prediction = model.predict(x_test[i]);
                int predictedClass = getPredictedClass(prediction);
                assertEquals(predictedClass, y_test_labels[i], "Prediction on unseen data is incorrect");
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
    }
