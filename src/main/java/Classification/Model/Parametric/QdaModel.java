package Classification.Model.Parametric;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.InstanceList.Partition;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Parameter.Parameter;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

public class QdaModel extends LdaModel implements Serializable {

    private HashMap<String, Matrix> W;

    /**
     * Training algorithm for the quadratic discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        String Ci;
        double determinant = 0, w0i;
        Matrix classCovariance, Wi;
        Vector averageVector, wi;
        HashMap<String, Double> w0 = new HashMap<>();
        HashMap<String, Vector> w = new HashMap<>();
        HashMap<String, Matrix> W = new HashMap<>();
        Partition classLists = new Partition(trainSet);
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        for (int i = 0; i < classLists.size(); i++) {
            Ci = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            classCovariance = classLists.get(i).covariance(averageVector);
            try {
                determinant = classCovariance.determinant();
                classCovariance.inverse();
            } catch (DeterminantZero | MatrixNotSquare ignored) {
            }
            Wi = classCovariance.clone();
            Wi.multiplyWithConstant(-0.5);
            W.put(Ci, Wi);
            try {
                wi = classCovariance.multiplyWithVectorFromLeft(averageVector);
                w.put(Ci, wi);
                w0i = -0.5 * (wi.dotProduct(averageVector) + Math.log(determinant)) + Math.log(priorDistribution.getProbability(Ci));
                w0.put(Ci, w0i);
            } catch (MatrixRowMismatch | VectorSizeMismatch ignored) {
            }
        }
        this.priorDistribution = priorDistribution;
        this.w = w;
        this.w0 = w0;
        this.W = W;
    }

    /**
     * Loads the Qda model from an input file.
     * @param fileName File name of the Qda model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            loadWandW0(input, size);
            W = new HashMap<>();
            for (int i = 0; i < size; i++){
                String c = input.readLine();
                Matrix matrix = loadMatrix(input);
                W.put(c, matrix);
            }
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs. It multiplies Matrix Wi with Vector xi
     * then calculates the dot product of it with xi. Then, again it finds the dot product of wi and xi and returns the summation with w0i.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The result of Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i.
     */
    protected double calculateMetric(Instance instance, String Ci) {
        double w0i;
        Vector xi, wi;
        Matrix Wi;
        xi = instance.toVector();
        Wi = W.get(Ci);
        wi = w.get(Ci);
        w0i = w0.get(Ci);
        try {
            return Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i;
        } catch (VectorSizeMismatch | MatrixRowMismatch vectorSizeMismatch) {
            return Double.MAX_VALUE;
        }
    }

    /**
     * Saves the Quadratic discriminant model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveWandW0(output);
            for (String c : W.keySet()){
                Matrix matrix = W.get(c);
                output.println(c);
                saveMatrix(output, matrix);
            }
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
