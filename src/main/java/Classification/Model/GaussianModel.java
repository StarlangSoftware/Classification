package Classification.Model;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;

import java.io.Serializable;
import Math.DiscreteDistribution;

public abstract class GaussianModel extends ValidatedModel implements Serializable{
    protected DiscreteDistribution priorDistribution;

    protected abstract double calculateMetric(Instance instance, String Ci);

    public String predict(Instance instance) {
        String predictedClass;
        String Ci;
        double metric;
        double maxMetric = -Double.MAX_VALUE;
        int size;
        if (instance instanceof CompositeInstance) {
            predictedClass = ((CompositeInstance)instance).getPossibleClassLabels().get(0);
            size = ((CompositeInstance)instance).getPossibleClassLabels().size();
        } else {
            predictedClass = priorDistribution.getMaxItem();
            size = priorDistribution.size();
        }
        for (int i = 0; i < size; i ++) {
            if (instance instanceof CompositeInstance) {
                Ci = ((CompositeInstance)instance).getPossibleClassLabels().get(i);
            } else {
                Ci = priorDistribution.getItem(i);
            }
            if (priorDistribution.containsItem(Ci)) {
                metric = calculateMetric(instance, Ci);
                if (metric > maxMetric){
                    maxMetric = metric;
                    predictedClass = Ci;
                }
            }
        }
        return predictedClass;
    }
}
