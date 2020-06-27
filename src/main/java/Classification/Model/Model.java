package Classification.Model;

import Classification.Instance.Instance;
import DataStructure.CounterHashMap;

import java.io.*;
import java.util.ArrayList;

public abstract class Model implements Serializable {

    /**
     * An abstract predict method that takes an {@link Instance} as an input.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The class label as a String.
     */
    public abstract String predict(Instance instance);

    /**
     * The save metohd takes a file name as an input and writes model to that file.
     *
     * @param fileName File name.
     */
    public void save(String fileName) {
        FileOutputStream outFile;
        ObjectOutputStream outObject;
        try {
            outFile = new FileOutputStream(fileName);
            outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Given an array of class labels, returns the maximum occurred one.
     *
     * @param classLabels An array of class labels.
     * @return The class label that occurs most in the array of class labels (mod of class label list).
     */
    public static String getMaximum(ArrayList<String> classLabels) {
        CounterHashMap<String> frequencies = new CounterHashMap<>();
        for (String label : classLabels) {
            frequencies.put(label);
        }
        return frequencies.max();
    }

}
