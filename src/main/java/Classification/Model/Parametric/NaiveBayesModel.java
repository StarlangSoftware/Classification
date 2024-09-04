package Classification.Model.Parametric;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.InstanceList.Partition;
import Classification.Parameter.Parameter;
import Math.Vector;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayesModel extends GaussianModel implements Serializable {
    private HashMap<String, Vector> classMeans = null;
    private HashMap<String, Vector> classDeviations = null;
    private HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions = null;

    /**
     * Training algorithm for Naive Bayes algorithm with a continuous data set.
     *
     * @param classLists        Instances are divided into K lists, where each list contains only instances from a single class
     */
    private void trainContinuousVersion(Partition classLists){
        String classLabel;
        classMeans = new HashMap<>();
        classDeviations = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classLabel = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            Vector averageVector = classLists.get(i).average().toVector();
            classMeans.put(classLabel, averageVector);
            Vector standardDeviationVector = classLists.get(i).standardDeviation().toVector();
            classDeviations.put(classLabel, standardDeviationVector);
        }
    }

    /**
     * Training algorithm for Naive Bayes algorithm with a discrete data set.
     * @param classLists Instances are divided into K lists, where each list contains only instances from a single class
     */
    private void trainDiscreteVersion(Partition classLists){
        classAttributeDistributions = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classAttributeDistributions.put(((InstanceListOfSameClass) classLists.get(i)).getClassLabel(), classLists.get(i).allAttributesDistribution());
        }
    }

    /**
     * Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data sets,
     * trainDiscreteVersion for discrete data sets.
     * @param trainSet Training data given to the algorithm
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        priorDistribution = trainSet.classDistribution();
        Partition classLists = new Partition(trainSet);
        if (classLists.get(0).get(0).getAttribute(0) instanceof DiscreteAttribute){
            trainDiscreteVersion(classLists);
        } else {
            trainContinuousVersion(classLists);
        }
    }

    /**
     * Loads the naive Bayes model from an input file.
     * @param fileName File name of the naive Bayes model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            classMeans = loadVectors(input, size);
            classDeviations = loadVectors(input, size);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
            if (si != 0){
                logLikelihood += -0.5 * Math.pow((xi - mi) / si, 2);
            }
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

    /**
     * Saves the Naive Bayes model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveVectors(output, classMeans);
            saveVectors(output, classDeviations);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
