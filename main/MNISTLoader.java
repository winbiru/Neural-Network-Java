package main;

import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {

    public static double[][] loadImages(String filePath) throws IOException {
        try (FileInputStream fis = new FileInputStream(filePath)) {
            byte[] header = new byte[16];
            fis.read(header);

            int numImages = byteArrayToInt(header, 4);
            int numRows = byteArrayToInt(header, 8);
            int numCols = byteArrayToInt(header, 12);

            double[][] images = new double[numImages][numRows * numCols];
            for (int i = 0; i < numImages; i++) {
                byte[] image = new byte[numRows * numCols];
                fis.read(image);
                for (int j = 0; j < image.length; j++) {
                    images[i][j] = (image[j] & 0xFF) / 255.0; // Normalize to [0, 1]
                }
            }
            return images;
        }
    }

    public static int[] loadLabels(String filePath) throws IOException {
        try (FileInputStream fis = new FileInputStream(filePath)) {
            byte[] header = new byte[8];
            fis.read(header);

            int numLabels = byteArrayToInt(header, 4);
            int[] labels = new int[numLabels];
            byte[] label = new byte[1];

            for (int i = 0; i < numLabels; i++) {
                fis.read(label);
                labels[i] = label[0] & 0xFF; // Convert byte to int
            }
            return labels;
        }
    }

    private static int byteArrayToInt(byte[] bytes, int offset) {
        return ((bytes[offset] & 0xFF) << 24) |
                ((bytes[offset + 1] & 0xFF) << 16) |
                ((bytes[offset + 2] & 0xFF) << 8) |
                (bytes[offset + 3] & 0xFF);
    }

    public static double[][] convertLabelsToOneHot(int[] labels, int numClasses) {
        double[][] oneHot = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            oneHot[i][labels[i]] = 1.0;
        }
        return oneHot;
    }
}
