package Classification.Model;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import DataStructure.CounterHashMap;
import Math.Matrix;
import Math.DiscreteDistribution;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public abstract class Model implements Serializable {

    /**
     * An abstract predict method that takes an {@link Instance} as an input.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The class label as a String.
     */
    public abstract String predict(Instance instance);

    public abstract HashMap<String, Double> predictProbability(Instance instance);

    public abstract void saveTxt(String fileName);

    /**
     * The save method takes a file name as an input and writes model to that file.
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
        } catch (IOException ignored) {
        }
    }

    /**
     * Saves an instance list to an output model file.
     * @param output Output model file
     * @param instanceList Instance list to be printed.
     */
    protected void saveInstanceList(PrintWriter output, InstanceList instanceList){
        Instance instance = instanceList.get(0);
        for (int i = 0; i < instance.attributeSize(); i++){
            if (instance.getAttribute(i) instanceof DiscreteAttribute){
                output.print("DISCRETE ");
            } else {
                if (instance.getAttribute(i) instanceof ContinuousAttribute){
                    output.print("CONTINUOUS ");
                }
            }
        }
        output.println();
        output.println(instanceList.size());
        for (int i = 0; i < instanceList.size(); i++){
            output.println(instanceList.get(i).toString());
        }
    }

    /**
     * Loads an instance list from an input model file.
     * @param input Input model file.
     * @return Instance list read from an input model file.
     * @throws IOException If the input model file can not be read, the method throws IOException.
     */
    protected InstanceList loadInstanceList(BufferedReader input) throws IOException {
        String[] types = input.readLine().split(" ");
        int instanceCount = Integer.parseInt(input.readLine());
        InstanceList instanceList = new InstanceList();
        for (int i = 0; i < instanceCount; i++){
            instanceList.add(loadInstance(input.readLine(), types));
        }
        return instanceList;
    }

    /**
     * Loads a single instance from a single line.
     * @param line Line containing the instance.
     * @param attributeTypes Type of the attributes of the instance. If th attribute is discrete, it is "DISCRETE",
     *                       otherwise it is "CONTINUOUS".
     * @return Instance read from the line.
     */
    protected Instance loadInstance(String line, String[] attributeTypes){
        String[] items = line.split(",");
        Instance instance = new Instance(items[items.length - 1]);
        for (int i = 0; i < items.length - 1; i++){
            switch (attributeTypes[i]){
                case "DISCRETE":
                    instance.addAttribute(items[i]);
                    break;
                case "CONTINUOUS":
                    instance.addAttribute(Double.parseDouble(items[i]));
                    break;
            }
        }
        return instance;
    }

    /**
     * Loads a discrete distribution from an input model file
     * @param input Input model file.
     * @return Discrete distribution read from an input model file.
     * @throws IOException If the input model file can not be read, the method throws IOException.
     */
    public static DiscreteDistribution loadDiscreteDistribution(BufferedReader input) throws IOException {
        DiscreteDistribution distribution = new DiscreteDistribution();
        int size = Integer.parseInt(input.readLine());
        for (int i = 0; i < size; i++){
            String line = input.readLine();
            String[] items = line.split(" ");
            int count = Integer.parseInt(items[1]);
            for(int j = 0; j < count; j++){
                distribution.addItem(items[0]);
            }
        }
        return distribution;
    }

    /**
     * Saves a discrete distribution to an output model file.
     * @param output Output model file.
     * @param distribution Discrete distribution to be printed.
     */
    public static void saveDiscreteDistribution(PrintWriter output, DiscreteDistribution distribution) {
        output.println(distribution.size());
        for (int i = 0; i < distribution.size(); i++){
            output.println(distribution.getItem(i) + " " + distribution.getValue(i));
        }
    }

    /**
     * Saves a matrix to an output model file.
     * @param output Output model file.
     * @param matrix Matrix to be printed.
     */
    protected void saveMatrix(PrintWriter output, Matrix matrix){
        output.println(matrix.getRow() + " " + matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++){
            output.print(matrix.getValue(i, 0));
            for (int j = 1; j < matrix.getColumn(); j++){
                output.print(" " + matrix.getValue(i, j));
            }
            output.println();
        }
    }

    /**
     * Loads a matrix from an input model file.
     * @param input Input model file.
     * @return Matrix read from the input model file.
     * @throws IOException If the input model file can not be read, the method throws IOException.
     */
    protected Matrix loadMatrix(BufferedReader input) throws IOException {
        String[] items = input.readLine().split(" ");
        Matrix matrix = new Matrix(Integer.parseInt(items[0]), Integer.parseInt(items[1]));
        for (int j = 0; j < matrix.getRow(); j++){
            String line = input.readLine();
            items = line.split(" ");
            for (int k = 0; k < matrix.getColumn(); k++){
                matrix.setValue(j, k, Double.parseDouble(items[k]));
            }
        }
        return matrix;
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
