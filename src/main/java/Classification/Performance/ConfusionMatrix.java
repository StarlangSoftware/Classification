package Classification.Performance;

import DataStructure.CounterHashMap;

import java.util.ArrayList;
import java.util.HashMap;

public class ConfusionMatrix {
    private HashMap<String, CounterHashMap<String>> matrix;
    private ArrayList<String> classLabels;

    /**
     * Constructor that sets the class labels {@link ArrayList} and creates new {@link HashMap} matrix
     * .
     *
     * @param classLabels {@link ArrayList} of String.
     */
    public ConfusionMatrix(ArrayList<String> classLabels) {
        this.classLabels = classLabels;
        matrix = new HashMap<>();
    }

    /**
     * The classify method takes two Strings; actual class and predicted class as inputs. If the matrix {@link HashMap} contains
     * given actual class String as a key, it then assigns the corresponding object of that key to a {@link CounterHashMap}, if not
     * it creates a new {@link CounterHashMap}. Then, it puts the given predicted class String to the counterHashMap and
     * also put this counterHashMap to the matrix {@link HashMap} together with the given actual class String.
     *
     * @param actualClass    String input actual class.
     * @param predictedClass String input predicted class.
     */
    public void classify(String actualClass, String predictedClass) {
        CounterHashMap<String> counterHashMap;
        if (matrix.containsKey(actualClass)) {
            counterHashMap = matrix.get(actualClass);
        } else {
            counterHashMap = new CounterHashMap<>();
        }
        counterHashMap.put(predictedClass);
        matrix.put(actualClass, counterHashMap);
    }

    /**
     * The addConfusionMatrix method takes a {@link ConfusionMatrix} as an input and loops through actual classes of that {@link HashMap}
     * and initially gets one row at a time. Then it puts the current row to the matrix {@link HashMap} together with the actual class string.
     *
     * @param confusionMatrix {@link ConfusionMatrix} input.
     */
    public void addConfusionMatrix(ConfusionMatrix confusionMatrix) {
        for (String actualClass : confusionMatrix.matrix.keySet()) {
            CounterHashMap<String> rowToBeAdded = confusionMatrix.matrix.get(actualClass);
            if (matrix.containsKey(actualClass)) {
                CounterHashMap<String> currentRow = matrix.get(actualClass);
                currentRow.add(rowToBeAdded);
                matrix.put(actualClass, currentRow);
            } else {
                matrix.put(actualClass, rowToBeAdded);
            }
        }
    }

    /**
     * The sumOfElements method loops through the keys in matrix {@link HashMap} and returns the summation of all the values of the keys.
     * I.e: TP+TN+FP+FN.
     *
     * @return The summation of values.
     */
    private double sumOfElements() {
        double result = 0;
        for (String actualClass : matrix.keySet()) {
            result += matrix.get(actualClass).sumOfCounts();
        }
        return result;
    }

    /**
     * The trace method loops through the keys in matrix {@link HashMap} and if the current key contains the actual key,
     * it accumulates the corresponding values. I.e: TP+TN.
     *
     * @return Summation of values.
     */
    private double trace() {
        double result = 0;
        for (String actualClass : matrix.keySet()) {
            if (matrix.get(actualClass).containsKey(actualClass)) {
                result += matrix.get(actualClass).get(actualClass);
            }
        }
        return result;
    }

    /**
     * The columnSum method takes a String predicted class as input, and loops through the keys in matrix {@link HashMap}.
     * If the current key contains the predicted class String, it accumulates the corresponding values. I.e: TP+FP.
     *
     * @param predictedClass String input predicted class.
     * @return Summation of values.
     */
    private double columnSum(String predictedClass) {
        double result = 0;
        for (String actualClass : matrix.keySet()) {
            if (matrix.get(actualClass).containsKey(predictedClass)) {
                result += matrix.get(actualClass).get(predictedClass);
            }
        }
        return result;
    }

    /**
     * The getAccuracy method returns the result of  TP+TN / TP+TN+FP+FN
     *
     * @return the result of  TP+TN / TP+TN+FP+FN
     */
    public double getAccuracy() {
        return trace() / sumOfElements();
    }

    /**
     * The precision method loops through the class labels and returns the resulting Array which has the result of TP/FP+TP.
     *
     * @return The result of TP/FP+TP.
     */
    public double[] precision() {
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++) {
            String actualClass = classLabels.get(i);
            if (matrix.containsKey(actualClass)) {
                result[i] = matrix.get(actualClass).get(actualClass) / columnSum(actualClass);
            }
        }
        return result;
    }

    /**
     * The recall method loops through the class labels and returns the resulting Array which has the result of TP/FN+TP.
     *
     * @return The result of TP/FN+TP.
     */
    public double[] recall() {
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++) {
            String actualClass = classLabels.get(i);
            if (matrix.containsKey(actualClass)) {
                result[i] = (matrix.get(actualClass).get(actualClass) + 0.0) / matrix.get(actualClass).sumOfCounts();
            }
        }
        return result;
    }

    /**
     * The fMeasure method loops through the class labels and returns the resulting Array which has the average of
     * recall and precision.
     *
     * @return The average of recall and precision.
     */
    public double[] fMeasure() {
        double[] precision = precision();
        double[] recall = recall();
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++) {
            result[i] = 2 / (1 / precision[i] + 1 / recall[i]);
        }
        return result;
    }

    /**
     * The weightedFMeasure method loops through the class labels and returns the resulting Array which has the weighted average of
     * recall and precision.
     *
     * @return The weighted average of recall and precision.
     */
    public double weightedFMeasure() {
        double[] fMeasure = fMeasure();
        double sum = 0;
        for (int i = 0; i < classLabels.size(); i++) {
            String actualClass = classLabels.get(i);
            sum += fMeasure[i] * matrix.get(actualClass).sumOfCounts();
        }
        return sum / sumOfElements();
    }

}
