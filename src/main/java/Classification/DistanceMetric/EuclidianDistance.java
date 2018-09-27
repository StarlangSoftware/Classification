package Classification.DistanceMetric;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;

import java.io.Serializable;

public class EuclidianDistance implements DistanceMetric, Serializable {

    public EuclidianDistance() {

    }

    /**
     * Calculates Euclidian distance between two instances. For continuous features: \sum_{i=1}^d (x_i^(1) - x_i^(2))^2,
     * For discrete features: \sum_{i=1}^d 1(x_i^(1) == x_i^(2))
     *
     * @param instance1 First instance
     * @param instance2 Second instance
     * @return Euclidian distance between two instances.
     */
    public double distance(Instance instance1, Instance instance2) {
        double result = 0;
        for (int i = 0; i < instance1.attributeSize(); i++) {
            if (instance1.getAttribute(i) instanceof DiscreteAttribute && instance2.getAttribute(i) instanceof DiscreteAttribute) {
                if (((DiscreteAttribute) instance1.getAttribute(i)).getValue() != null && ((DiscreteAttribute) instance1.getAttribute(i)).getValue().compareTo(((DiscreteAttribute) instance2.getAttribute(i)).getValue()) != 0) {
                    result += 1;
                }
            } else {
                if (instance1.getAttribute(i) instanceof ContinuousAttribute && instance2.getAttribute(i) instanceof ContinuousAttribute) {
                    result += Math.pow(((ContinuousAttribute) instance1.getAttribute(i)).getValue() - ((ContinuousAttribute) instance2.getAttribute(i)).getValue(), 2);
                }
            }
        }
        return result;
    }
}
