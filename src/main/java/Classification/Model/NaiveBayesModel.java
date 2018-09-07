package Classification.Model;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;
import Math.Vector;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayesModel extends GaussianModel implements Serializable{
    private HashMap<String, Vector> classMeans = null;
    private HashMap<String, Vector> classDeviations = null;
    private HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions = null;

    @Override
    protected double calculateMetric(Instance instance, String Ci) {
        if (classAttributeDistributions == null){
            return logLikelihoodContinuous(Ci, instance);
        } else {
            return logLikelihoodDiscrete(Ci, instance);
        }
    }

    private double logLikelihoodContinuous(String classLabel, Instance instance){
        double xi, mi, si;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        for (int i = 0; i < instance.attributeSize(); i++){
            xi = ((ContinuousAttribute) instance.getAttribute(i)).getValue();
            mi = classMeans.get(classLabel).getValue(i);
            si = classDeviations.get(classLabel).getValue(i);
            logLikelihood += -0.5 * Math.pow((xi - mi) / si, 2);
        }
        return logLikelihood;
    }

    private double logLikelihoodDiscrete(String classLabel, Instance instance){
        String xi;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        ArrayList<DiscreteDistribution> attributeDistributions = classAttributeDistributions.get(classLabel);
        for (int i = 0; i < instance.attributeSize(); i++){
            xi = ((DiscreteAttribute) instance.getAttribute(i)).getValue();
            logLikelihood += Math.log(attributeDistributions.get(i).getProbabilityLaplaceSmoothing(xi));
        }
        return logLikelihood;
    }

    public NaiveBayesModel(DiscreteDistribution priorDistribution, HashMap<String, Vector> classMeans, HashMap<String, Vector> classDeviations){
        this.priorDistribution = priorDistribution;
        this.classMeans = classMeans;
        this.classDeviations = classDeviations;
    }

    public NaiveBayesModel(DiscreteDistribution priorDistribution, HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions){
        this.priorDistribution = priorDistribution;
        this.classAttributeDistributions = classAttributeDistributions;
    }

}
