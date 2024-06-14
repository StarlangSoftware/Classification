package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.KMeansModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.KMeansParameter;
import Classification.InstanceList.Partition;
import Math.DiscreteDistribution;

public class KMeans extends Classifier {

    /**
     * Training algorithm for K-Means classifier. K-Means finds the mean of each class for training.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters distance metric used to calculate the distance between two instances.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        InstanceList classMeans = new InstanceList();
        Partition classLists = new Partition(trainSet);
        for (int i = 0; i < classLists.size(); i++) {
            classMeans.add(classLists.get(i).average());
        }
        model = new KMeansModel(priorDistribution, classMeans, ((KMeansParameter) parameters).getDistanceMetric());
    }

    /**
     * Loads the K-means model from an input file.
     * @param fileName File name of the K-means model.
     */
    @Override
    public void loadModel(String fileName) {
        model = new KMeansModel(fileName);
    }
}
