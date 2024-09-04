package Classification.Model.Parametric;

import Classification.DistanceMetric.DistanceMetric;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Parameter.KMeansParameter;
import Classification.Parameter.Parameter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

public class KMeansModel extends GaussianModel implements Serializable {
    private InstanceList classMeans;
    private DistanceMetric distanceMetric;

    /**
     * Training algorithm for K-Means classifier. K-Means finds the mean of each class for training.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters distance metric used to calculate the distance between two instances.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        priorDistribution = trainSet.classDistribution();
        classMeans = new InstanceList();
        Partition classLists = new Partition(trainSet);
        for (int i = 0; i < classLists.size(); i++) {
            classMeans.add(classLists.get(i).average());
        }
        this.distanceMetric = ((KMeansParameter) parameters).getDistanceMetric();
    }

    /**
     * Loads the K-means model from an input file.
     * @param fileName File name of the K-means model.
     */
    @Override
    public void loadModel(String fileName) {
        this.distanceMetric = new EuclidianDistance();
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            loadPriorDistribution(input);
            classMeans = loadInstanceList(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means, if
     * the corresponding class label is same as the given String it returns the negated distance between given instance and the
     * current item of class means. Otherwise it returns the smallest negative number.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The negated distance between given instance and the current item of class means.
     */
    protected double calculateMetric(Instance instance, String Ci) {
        for (int i = 0; i < classMeans.size(); i++) {
            if (classMeans.get(i).getClassLabel().equals(Ci)) {
                return -distanceMetric.distance(instance, classMeans.get(i));
            }
        }
        return -Double.MAX_VALUE;
    }

    /**
     * Saves the K-Means model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveInstanceList(output, classMeans);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
