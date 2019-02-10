package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.KMeansModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.RocchioParameter;
import Classification.InstanceList.Partition;
import Math.DiscreteDistribution;

public class KMeans extends Classifier {

    /**
     * Training algorithm for Rocchio classifier. Rocchio finds the mean of each class for training.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters distanceMetric: distance metric used to calculate the distance between two instances.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        InstanceList classMeans = new InstanceList();
        Partition classLists = trainSet.divideIntoClasses();
        for (int i = 0; i < classLists.size(); i++) {
            classMeans.add(classLists.get(i).average());
        }
        model = new KMeansModel(priorDistribution, classMeans, ((RocchioParameter) parameters).getDistanceMetric());
    }
}
