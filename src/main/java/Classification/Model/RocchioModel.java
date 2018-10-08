package Classification.Model;

import Classification.DistanceMetric.DistanceMetric;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.Serializable;

public class RocchioModel extends GaussianModel implements Serializable {
    private InstanceList classMeans;
    private DistanceMetric distanceMetric;

    /**
     * The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.
     *
     * @param priorDistribution {@link DiscreteDistribution} input.
     * @param classMeans        {@link InstanceList} of class means.
     * @param distanceMetric    {@link DistanceMetric} input.
     */
    public RocchioModel(DiscreteDistribution priorDistribution, InstanceList classMeans, DistanceMetric distanceMetric) {
        this.classMeans = classMeans;
        this.priorDistribution = priorDistribution;
        this.distanceMetric = distanceMetric;
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

}
