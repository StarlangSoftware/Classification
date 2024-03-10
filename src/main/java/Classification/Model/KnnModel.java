package Classification.Model;

import Classification.DistanceMetric.DistanceMetric;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

public class KnnModel extends Model implements Serializable {

    private final InstanceList data;
    private final int k;
    private final DistanceMetric distanceMetric;

    /**
     * Constructor that sets the data {@link InstanceList}, k value and the {@link DistanceMetric}.
     *
     * @param data           {@link InstanceList} input.
     * @param k              K value.
     * @param distanceMetric {@link DistanceMetric} input.
     */
    public KnnModel(InstanceList data, int k, DistanceMetric distanceMetric) {
        this.data = data;
        this.k = k;
        this.distanceMetric = distanceMetric;
    }

    public KnnModel(String fileName){
        this.distanceMetric = new EuclidianDistance();
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            k = Integer.parseInt(input.readLine());
            data = loadInstanceList(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The predict method takes an {@link Instance} as an input and finds the nearest neighbors of given instance. Then
     * it returns the first possible class label as the predicted class.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The first possible class label as the predicted class.
     */
    public String predict(Instance instance) {
        InstanceList nearestNeighbors = nearestNeighbors(instance);
        String predictedClass;
        if (instance instanceof CompositeInstance && nearestNeighbors.size() == 0) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
        } else {
            predictedClass = Model.getMaximum(nearestNeighbors.getClassLabels());
        }
        return predictedClass;
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        InstanceList nearestNeighbors = nearestNeighbors(instance);
        return nearestNeighbors.classDistribution().getProbabilityDistribution();
    }

    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            output.println(k);
            saveInstanceList(output, data);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }


    public void generateTestCode(String codeFileName, String methodName, String inputFileName){
        try {
            int attributeCount = data.get(0).attributeSize();
            PrintWriter output = new PrintWriter(codeFileName);
            output.println("public static String " + methodName + "(String[] testData) throws FileNotFoundException{");
            output.println("\tCounterHashMap<String> counts = new CounterHashMap<>();");
            output.println("\tString[][] trainData = new String[" + data.size() + "][" + (attributeCount + 1) + "];");
            output.println("\tScanner input = new Scanner(new File(\"" + inputFileName + "\"));");
            output.println("\tfor (int i = 0; i < " + data.size() + "; i++){");
            output.println("\t\tString[] items = input.nextLine().split(\" \");");
            output.println("\t\tfor (int j = 0; j < " + (attributeCount + 1) + "; j++){");
            output.println("\t\t\ttrainData[i][j] = items[j];");
            output.println("\t\t}");
            output.println("\t}");
            output.println("\tinput.close();");
            output.println("\tint minDistance = " + attributeCount + ";");
            output.println("\tfor (int i = 0; i < " + data.size() + "; i++){");
            output.println("\t\tint count = 0;");
            output.println("\t\tfor (int j = 0; j < " + attributeCount + "; j++){");
            output.println("\t\t\tif (!testData[j].equals(trainData[i][j])){");
            output.println("\t\t\t\tcount++;");
            output.println("\t\t\t}");
            output.println("\t\t}");
            output.println("\t\tif (count < minDistance){");
            output.println("\t\t\tminDistance = count;");
            output.println("\t\t}");
            output.println("\t}");
            output.println("\tfor (int i = 0; i < " + data.size() + "; i++){");
            output.println("\t\tint count = 0;");
            output.println("\t\tfor (int j = 0; j < " + attributeCount + "; j++){");
            output.println("\t\t\tif (!testData[j].equals(trainData[i][j])){");
            output.println("\t\t\t\tcount++;");
            output.println("\t\t\t}");
            output.println("\t\t}");
            output.println("\t\tif (count == minDistance){");
            output.println("\t\t\tcounts.put(trainData[i][" + attributeCount + "]);");
            output.println("\t\t}");
            output.println("\t}");
            output.println("\treturn counts.max();");
            output.println("}");
            output.close();
        } catch (FileNotFoundException ignored) {
        }
    }

    /**
     * The nearestNeighbors method takes an {@link Instance} as an input. First it gets the possible class labels, then loops
     * through the data {@link InstanceList} and creates new {@link ArrayList} of {@link KnnInstance}s and adds the corresponding data with
     * the distance between data and given instance. After sorting this newly created ArrayList, it loops k times and
     * returns the first k instances as an {@link InstanceList}.
     *
     * @param instance {@link Instance} to find nearest neighbors/
     * @return The first k instances which are nearest to the given instance as an {@link InstanceList}.
     */
    public InstanceList nearestNeighbors(Instance instance) {
        InstanceList result = new InstanceList();
        ArrayList<KnnInstance> instances = new ArrayList<>();
        ArrayList<String> possibleClassLabels = null;
        if (instance instanceof CompositeInstance) {
            possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
        }
        for (int i = 0; i < data.size(); i++) {
            if (!(instance instanceof CompositeInstance) || possibleClassLabels.contains(data.get(i).getClassLabel())) {
                instances.add(new KnnInstance(data.get(i), distanceMetric.distance(data.get(i), instance)));
            }
        }
        instances.sort(new KnnInstanceComparator());
        for (int i = 0; i < Math.min(k, instances.size()); i++) {
            result.add(instances.get(i).instance);
        }
        return result;
    }


    private static class KnnInstance {
        private final double distance;
        private final Instance instance;

        /**
         * The constructor that sets the instance and distance value.
         *
         * @param instance {@link Instance} input.
         * @param distance Double distance value.
         */
        private KnnInstance(Instance instance, double distance) {
            this.instance = instance;
            this.distance = distance;
        }

        /**
         * The toString method returns the concatenation of class label of the instance and the distance value.
         *
         * @return The concatenation of class label of the instance and the distance value.
         */
        public String toString() {
            String str = "";
            str += instance.getClassLabel() + " " + distance;
            return str;
        }
    }


    private static class KnnInstanceComparator implements Comparator<KnnInstance> {
        /**
         * The compare method takes two {@link KnnInstance}s as inputs and returns -1 if the distance of first instance is
         * less than the distance of second instance, 1 if the distance of first instance is greater than the distance of second instance,
         * and 0 if they are equal to each other.
         *
         * @param instance1 First {@link KnnInstance} to compare.
         * @param instance2 SEcond {@link KnnInstance} to compare.
         * @return -1 if the distance of first instance is less than the distance of second instance,
         * 1 if the distance of first instance is greater than the distance of second instance,
         * 0 if they are equal to each other.
         */
        public int compare(KnnInstance instance1, KnnInstance instance2) {
            return Double.compare(instance1.distance, instance2.distance);
        }
    }
}
