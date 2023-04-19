package Classification.Model;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.DistanceMetric.DistanceMetric;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class KMeansModel extends GaussianModel implements Serializable {
    private InstanceList classMeans;
    private DistanceMetric distanceMetric;

    /**
     * The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.
     *
     * @param priorDistribution {@link DiscreteDistribution} input.
     * @param classMeans        {@link InstanceList} of class means.
     * @param distanceMetric    {@link DistanceMetric} input.
     */
    public KMeansModel(DiscreteDistribution priorDistribution, InstanceList classMeans, DistanceMetric distanceMetric) {
        this.classMeans = classMeans;
        this.priorDistribution = priorDistribution;
        this.distanceMetric = distanceMetric;
    }

    public KMeansModel(String fileName){
        this.distanceMetric = new EuclidianDistance();
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), StandardCharsets.UTF_8));
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
