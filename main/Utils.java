package main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class Utils {
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
    //Hàm này tạo ra một số ngẫu nhiên sử dụng phương pháp Box-Muller.
    public static double randn() {
        // Create random in range [0, 1)
        double u1 = Math.random();
        double u2 = Math.random();

        // Box-Muller Transform
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    //Hàm này chuyển vị mảng đã cho.
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
    //Hàm này khởi tạo trọng số sử dụng phương pháp khởi tạo He.
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
    // Hàm này khởi tạo trọng số sử dụng phương pháp khởi tạo Xavier.
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
    //Hàm này áp dụng hàm kích hoạt sigmoid cho mỗi phần tử trong mảng đã cho.
    public static double[] sigmoid(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = 1.0 / (1 + Math.exp(-arr[i]));
        }
        return result;
    }
    //Hàm này tính đạo hàm của hàm kích hoạt sigmoid cho mỗi phần tử trong mảng đã cho.
    public static double[] sigmoidDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] * (1 - arr[i]);
        }
        return result;
    }
    //Hàm này áp dụng hàm kích hoạt ReLU cho mỗi phần tử trong mảng đã cho.
    public static double[] relu(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = Math.max(0, arr[i]);
        }
        return result;
    }
    //Hàm này tính đạo hàm của hàm kích hoạt ReLU cho mỗi phần tử trong mảng đã cho.
    public static double[] reluDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? 1 : 0;
        }
        return result;
    }
    //Hàm này áp dụng hàm kích hoạt Leaky ReLU cho mỗi phần tử trong mảng đã cho.
    public static double[] leakyRelu(double[] arr, double alpha) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? arr[i] : alpha * arr[i];
        }
        return result;
    }
    //Hàm này tính đạo hàm của hàm kích hoạt Leaky ReLU cho mỗi phần tử trong mảng đã cho.
    public static double[] leakyReluDerivative(double[] arr) {
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i] > 0 ? 1 : 0.01;
        }
        return result;
    }
    //Hàm này áp dụng hàm kích hoạt softmax cho mỗi phần tử trong mảng đã cho.
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
    //Hàm này tính đạo hàm của hàm kích hoạt softmax cho mỗi phần tử trong mảng đã cho.
    public static double[] softmaxDerivative(double[] arr) {
        double[] s = softmax(arr);
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = s[i] * (1 - s[i]);
        }
        return result;
    }
    //Hàm này tính giá trị hàm mất bình phương trung bình (MSE) giữa các giá trị dự đoán và giá trị thực tế.
    public static double mse(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += Math.pow(y_pred[i] - y_true[i], 2);
        }
        return sum / y_pred.length;
    }
    //Hàm này tính đạo hàm của hàm mất bình phương trung bình (MSE) với giá trị dự đoán đã cho.
    public static double[] mseGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = 2 * (y_pred[i] - y_true[i]) / y_pred.length;
        }
        return gradients;
    }

    //Hàm này tính giá trị hàm mất phân loại đa nhãn (Categorical Cross-Entropy) giữa các giá trị dự đoán và giá trị thực tế.
    public static double categoricalCrossEntropy(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += -y_true[i] * Math.log(y_pred[i] + 1e-15);
        }
        return sum;
    }
    //Hàm này tính đạo hàm của hàm mất phân loại đa nhãn (Categorical Cross-Entropy) với giá trị dự đoán đã cho.
    public static double[] categoricalCrossEntropyGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = y_pred[i] - y_true[i];
        }
        return gradients;
    }
    //Hàm này tính giá trị hàm mất nhị phân (Binary Cross-Entropy) giữa các giá trị dự đoán và giá trị thực tế.
    public static double binaryCrossEntropy(double[] y_pred, double[] y_true) {
        double sum = 0;
        for (int i = 0; i < y_pred.length; i++) {
            sum += y_true[i] * Math.log(y_pred[i] + 1e-15) + (1 - y_true[i]) * Math.log(1 - y_pred[i] + 1e-15);
        }
        return -sum / y_pred.length;
    }
    //Hàm này tính đạo hàm của hàm mất nhị phân (Binary Cross-Entropy) với giá trị dự đoán đã cho.
    public static double[] binaryCrossEntropyGradient(double[] y_pred, double[] y_true) {
        double[] gradients = new double[y_pred.length];
        for (int i = 0; i < y_pred.length; i++) {
            gradients[i] = (y_pred[i] - y_true[i]) / ((y_pred[i] + 1e-15) * (1 - y_pred[i] + 1e-15));
        }
        return gradients;
    }
    //Hàm này ghi dữ liệu vào tệp có đường dẫn đã cho.
    public static void writeFile(String path, String data) throws IOException {
        Files.write(Paths.get(path), data.getBytes());
    }
    // Hàm này đọc nội dung của tệp có đường dẫn đã cho.
    public static String readFile(String path) throws IOException {
        return new String(Files.readAllBytes(Paths.get(path)));
    }
    // Hàm này chuyển đổi các nhãn thành các vector one-hot.
    public static double[][] convertToOneHot(int[] labels, int numClasses) {
        double[][] oneHotVectors = new double[labels.length][numClasses];

        for (int i = 0; i < labels.length; i++) {
            oneHotVectors[i][labels[i]] = 1;
        }

        return oneHotVectors;
    }
    // Hàm này tiền xử lý các hình ảnh bằng cách chuẩn hóa giá trị của các điểm ảnh thành khoảng [0, 1].
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
