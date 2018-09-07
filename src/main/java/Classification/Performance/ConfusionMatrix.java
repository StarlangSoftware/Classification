package Classification.Performance;

import DataStructure.CounterHashMap;

import java.util.ArrayList;
import java.util.HashMap;

public class ConfusionMatrix {
    private HashMap<String, CounterHashMap<String>> matrix;
    private ArrayList<String> classLabels;

    public ConfusionMatrix(ArrayList<String> classLabels){
        this.classLabels = classLabels;
        matrix = new HashMap<>();
    }

    public void classify(String actualClass, String predictedClass){
        CounterHashMap<String> counterHashMap;
        if (matrix.containsKey(actualClass)){
            counterHashMap = matrix.get(actualClass);
        } else {
            counterHashMap = new CounterHashMap<>();
        }
        counterHashMap.put(predictedClass);
        matrix.put(actualClass, counterHashMap);
    }

    public void addConfusionMatrix(ConfusionMatrix confusionMatrix){
        for (String actualClass : confusionMatrix.matrix.keySet()){
            CounterHashMap<String> rowToBeAdded = confusionMatrix.matrix.get(actualClass);
            if (matrix.containsKey(actualClass)){
                CounterHashMap<String> currentRow = matrix.get(actualClass);
                currentRow.add(rowToBeAdded);
                matrix.put(actualClass, currentRow);
            } else {
                matrix.put(actualClass, rowToBeAdded);
            }
        }
    }

    private double sumOfElements(){
        double result = 0;
        for (String actualClass : matrix.keySet()){
            result += matrix.get(actualClass).sumOfCounts();
        }
        return result;
    }

    private double trace(){
        double result = 0;
        for (String actualClass : matrix.keySet()){
            if (matrix.get(actualClass).containsKey(actualClass)){
                result += matrix.get(actualClass).get(actualClass);
            }
        }
        return result;
    }

    private double columnSum(String predictedClass){
        double result = 0;
        for (String actualClass : matrix.keySet()){
            if (matrix.get(actualClass).containsKey(predictedClass)){
                result += matrix.get(actualClass).get(predictedClass);
            }
        }
        return result;
    }

    public double getAccuracy(){
        return trace() / sumOfElements();
    }

    public double[] precision(){
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++){
            String actualClass = classLabels.get(i);
            if (matrix.containsKey(actualClass)){
                result[i] = matrix.get(actualClass).get(actualClass) / columnSum(actualClass);
            }
        }
        return result;
    }

    public double[] recall(){
        double[] result = new double[classLabels.size()];
        for (int i = 0; i < classLabels.size(); i++){
            String actualClass = classLabels.get(i);
            if (matrix.containsKey(actualClass)){
                result[i] = (matrix.get(actualClass).get(actualClass) + 0.0) / matrix.get(actualClass).sumOfCounts();
            }
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
            String actualClass = classLabels.get(i);
            sum += fMeasure[i] * matrix.get(actualClass).sumOfCounts();
        }
        return sum / sumOfElements();
    }

}
