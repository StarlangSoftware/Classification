package Classification.Model;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;
import Math.Vector;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayesModel extends GaussianModel implements Serializable {
    private HashMap<String, Vector> classMeans = null;
    private HashMap<String, Vector> classDeviations = null;
    private HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions = null;

    /**
     * A constructor that sets the priorDistribution, classMeans and classDeviations.
     *
     * @param priorDistribution {@link DiscreteDistribution} input.
     * @param classMeans        A {@link HashMap} of String and {@link Vector}.
     * @param classDeviations   A {@link HashMap} of String and {@link Vector}.
     */
    public NaiveBayesModel(DiscreteDistribution priorDistribution, HashMap<String, Vector> classMeans, HashMap<String, Vector> classDeviations) {
        this.priorDistribution = priorDistribution;
        this.classMeans = classMeans;
        this.classDeviations = classDeviations;
    }

    /**
     * A constructor that sets the priorDistribution and classAttributeDistributions.
     *
     * @param priorDistribution           {@link DiscreteDistribution} input.
     * @param classAttributeDistributions {@link HashMap} of String and {@link ArrayList} of {@link DiscreteDistribution}s.
     */
    public NaiveBayesModel(DiscreteDistribution priorDistribution, HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions) {
        this.priorDistribution = priorDistribution;
        this.classAttributeDistributions = classAttributeDistributions;
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs and it returns the log likelihood of
     * these inputs.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The log likelihood of inputs.
     */
    @Override
    protected double calculateMetric(Instance instance, String Ci) {
        if (classAttributeDistributions == null) {
            return logLikelihoodContinuous(Ci, instance);
        } else {
            return logLikelihoodDiscrete(Ci, instance);
        }
    }

    /**
     * The logLikelihoodContinuous method takes an {@link Instance} and a class label as inputs. First it gets the logarithm
     * of given class label's probability via prior distribution as logLikelihood. Then it loops times of given instance attribute size, and accumulates the
     * logLikelihood by calculating -0.5 * ((xi - mi) / si )** 2).
     *
     * @param classLabel String input class label.
     * @param instance   {@link Instance} input.
     * @return The log likelihood of given class label and {@link Instance}.
     */
    private double logLikelihoodContinuous(String classLabel, Instance instance) {
        double xi, mi, si;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        for (int i = 0; i < instance.attributeSize(); i++) {
            xi = ((ContinuousAttribute) instance.getAttribute(i)).getValue();
            mi = classMeans.get(classLabel).getValue(i);
            si = classDeviations.get(classLabel).getValue(i);
            logLikelihood += -0.5 * Math.pow((xi - mi) / si, 2);
        }
        return logLikelihood;
    }

    /**
     * The logLikelihoodDiscrete method takes an {@link Instance} and a class label as inputs. First it gets the logarithm
     * of given class label's probability via prior distribution as logLikelihood and gets the class attribute distribution of given class label.
     * Then it loops times of given instance attribute size, and accumulates the logLikelihood by calculating the logarithm of
     * corresponding attribute distribution's smoothed probability by using laplace smoothing on xi.
     *
     * @param classLabel String input class label.
     * @param instance   {@link Instance} input.
     * @return The log likelihood of given class label and {@link Instance}.
     */
    private double logLikelihoodDiscrete(String classLabel, Instance instance) {
        String xi;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        ArrayList<DiscreteDistribution> attributeDistributions = classAttributeDistributions.get(classLabel);
        for (int i = 0; i < instance.attributeSize(); i++) {
            xi = ((DiscreteAttribute) instance.getAttribute(i)).getValue();
            logLikelihood += Math.log(attributeDistributions.get(i).getProbabilityLaplaceSmoothing(xi));
        }
        return logLikelihood;
    }
}
