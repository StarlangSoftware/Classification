package Classification.Model;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;

import java.io.Serializable;
import java.util.HashMap;

import Math.DiscreteDistribution;

public abstract class GaussianModel extends ValidatedModel implements Serializable {
    protected DiscreteDistribution priorDistribution;

    /**
     * Abstract method calculateMetric takes an {@link Instance} and a String as inputs.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return A double value as metric.
     */
    protected abstract double calculateMetric(Instance instance, String Ci);

    /**
     * The predict method takes an Instance as an input. First it gets the size of prior distribution and loops this size times.
     * Then it gets the possible class labels and and calculates metric value. At the end, it returns the class which has the
     * maximum value of metric.
     *
     * @param instance {@link Instance} to predict.
     * @return The class which has the maximum value of metric.
     */
    public String predict(Instance instance) {
        String predictedClass;
        String Ci;
        double metric;
        double maxMetric = -Double.MAX_VALUE;
        int size;
        if (instance instanceof CompositeInstance) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
            size = ((CompositeInstance) instance).getPossibleClassLabels().size();
        } else {
            predictedClass = priorDistribution.getMaxItem();
            size = priorDistribution.size();
        }
        for (int i = 0; i < size; i++) {
            if (instance instanceof CompositeInstance) {
                Ci = ((CompositeInstance) instance).getPossibleClassLabels().get(i);
            } else {
                Ci = priorDistribution.getItem(i);
            }
            if (priorDistribution.containsItem(Ci)) {
                metric = calculateMetric(instance, Ci);
                if (metric > maxMetric) {
                    maxMetric = metric;
                    predictedClass = Ci;
                }
            }
        }
        return predictedClass;
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return null;
    }
}
