package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.InstanceList.Partition;
import Classification.Parameter.Parameter;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

public class LdaModel extends GaussianModel implements Serializable {

    protected HashMap<String, Double> w0;
    protected HashMap<String, Vector> w;

    /**
     * Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        String Ci;
        double w0i;
        Matrix covariance, classCovariance;
        Vector averageVector, wi;
        HashMap<String, Double> w0 = new HashMap<>();
        HashMap<String, Vector> w = new HashMap<>();
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        Partition classLists = new Partition(trainSet);
        covariance = new Matrix(trainSet.get(0).continuousAttributeSize(), trainSet.get(0).continuousAttributeSize());
        for (int i = 0; i < classLists.size(); i++) {
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            classCovariance = classLists.get(i).covariance(averageVector);
            classCovariance.multiplyWithConstant(classLists.get(i).size() - 1);
            try {
                covariance.add(classCovariance);
            } catch (MatrixDimensionMismatch ignored) {
            }
        }
        covariance.divideByConstant(trainSet.size() - classLists.size());
        try {
            covariance.inverse();
        } catch (DeterminantZero | MatrixNotSquare ignored) {
        }
        for (int i = 0; i < classLists.size(); i++) {
            Ci = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            try {
                wi = covariance.multiplyWithVectorFromRight(averageVector);
                w.put(Ci, wi);
                w0i = -0.5 * wi.dotProduct(averageVector) + Math.log(priorDistribution.getProbability(Ci));
                w0.put(Ci, w0i);
            } catch (MatrixColumnMismatch | VectorSizeMismatch ignored) {
            }
        }
        this.priorDistribution = priorDistribution;
        this.w = w;
        this.w0 = w0;
    }

    /**
     * Loads the Lda model from an input file.
     * @param fileName File name of the Lda model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            loadWandW0(input, size);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Loads w0 and w hash maps from an input file. The number of items in the hash map is given by the parameter size.
     * @param input Input file
     * @param size Number of items in the hash map read.
     * @throws IOException If the file can not be read, it throws IOException.
     */
    protected void loadWandW0(BufferedReader input, int size) throws IOException {
        w0 = new HashMap<>();
        for (int i = 0; i < size; i++){
            String line = input.readLine();
            String[] items = line.split(" ");
            w0.put(items[0], Double.parseDouble(items[1]));
        }
        w = loadVectors(input, size);
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs. It returns the dot product of given Instance
     * and wi plus w0i.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The dot product of given Instance and wi plus w0i.
     */
    @Override
    protected double calculateMetric(Instance instance, String Ci) {
        double w0i;
        Vector xi, wi;
        xi = instance.toVector();
        wi = w.get(Ci);
        w0i = w0.get(Ci);
        try {
            return wi.dotProduct(xi) + w0i;
        } catch (VectorSizeMismatch vectorSizeMismatch) {
            return Double.MAX_VALUE;
        }
    }

    /**
     * Saves w and w0 hash maps to an output file
     * @param output Output file
     */
    protected void saveWandW0(PrintWriter output){
        for (String c : w0.keySet()){
            output.println(c + " " + w0.get(c));
        }
        saveVectors(output, w);
    }

    /**
     * Saves the Linear discriminant analysis model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveWandW0(output);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
