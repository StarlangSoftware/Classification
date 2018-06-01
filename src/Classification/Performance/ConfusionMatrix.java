package Classification.Performance;

import Math.*;

import java.util.ArrayList;

public class ConfusionMatrix {
    private Matrix matrix;
    private ArrayList<String> classLabels;

    public ConfusionMatrix(ArrayList<String> classLabels){
        this.classLabels = classLabels;
        matrix = new Matrix(classLabels.size(), classLabels.size());
    }

    public void classify(String actualClass, String predictedClass){
        int actual, predicted;
        actual = classLabels.indexOf(actualClass);
        predicted = classLabels.indexOf(predictedClass);
        if (actual != -1 && predicted != -1){
            matrix.increment(actual, predicted);
        }
    }

    public void addConfusionMatrix(ConfusionMatrix confusionMatrix){
        try {
            matrix.add(confusionMatrix.matrix);
        } catch (MatrixDimensionMismatch matrixDimensionMismatch) {
        }
    }

    public double getAccuracy(){
        return matrix.trace() / matrix.sumOfElements();
    }

    public double[] precision(){
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++){
            result[i] = matrix.getValue(i, i) / matrix.columnSum(i);
        }
        return result;
    }

    public double[] recall(){
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++){
            result[i] = matrix.getValue(i, i) / matrix.rowSum(i);
        }
        return result;
    }

    public double[] fMeasure(){
        double[] precision = precision();
        double[] recall = recall();
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++){
            result[i] = 2 / (1 / precision[i] + 1 / recall[i]);
        }
        return result;
    }

    public double weightedFMeasure(){
        double[] fMeasure = fMeasure();
        double sum = 0;
        for (int i = 0; i < classLabels.size(); i++){
            sum += fMeasure[i] * matrix.rowSum(i);
        }
        return sum / matrix.sumOfElements();
    }

}
