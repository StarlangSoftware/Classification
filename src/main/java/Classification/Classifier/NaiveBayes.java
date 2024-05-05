package Classification.Classifier;

import Classification.Attribute.DiscreteAttribute;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.Model.NaiveBayesModel;
import Classification.Parameter.Parameter;
import Classification.InstanceList.Partition;
import Math.Vector;
import Math.DiscreteDistribution;

import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayes extends Classifier{
    /**
     * Training algorithm for Naive Bayes algorithm with a continuous data set.
     *
     * @param priorDistribution Probability distribution of classes P(C_i)
     * @param classLists        Instances are divided into K lists, where each list contains only instances from a single class
     */
    private void trainContinuousVersion(DiscreteDistribution priorDistribution, Partition classLists){
        String classLabel;
        HashMap<String, Vector> classMeans = new HashMap<>();
        HashMap<String, Vector> classDeviations = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classLabel = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            Vector averageVector = classLists.get(i).average().toVector();
            classMeans.put(classLabel, averageVector);
            Vector standardDeviationVector = classLists.get(i).standardDeviation().toVector();
            classDeviations.put(classLabel, standardDeviationVector);
        }
        model = new NaiveBayesModel(priorDistribution, classMeans, classDeviations);
    }

    /**
     * Training algorithm for Naive Bayes algorithm with a discrete data set.
     * @param priorDistribution Probability distribution of classes P(C_i)
     * @param classLists Instances are divided into K lists, where each list contains only instances from a single class
     */
    private void trainDiscreteVersion(DiscreteDistribution priorDistribution, Partition classLists){
        HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classAttributeDistributions.put(((InstanceListOfSameClass) classLists.get(i)).getClassLabel(), classLists.get(i).allAttributesDistribution());
        }
        model = new NaiveBayesModel(priorDistribution, classAttributeDistributions);
    }

    /**
     * Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data sets,
     * trainDiscreteVersion for discrete data sets.
     * @param trainSet Training data given to the algorithm
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        Partition classLists = new Partition(trainSet);
        if (classLists.get(0).get(0).getAttribute(0) instanceof DiscreteAttribute){
            trainDiscreteVersion(priorDistribution, classLists);
        } else {
            trainContinuousVersion(priorDistribution, classLists);
        }
    }

    /**
     * Loads the naive Bayes model from an input file.
     * @param fileName File name of the naive Bayes model.
     */
    @Override
    public void loadModel(String fileName) {
        model = new NaiveBayesModel(fileName);
    }
}
