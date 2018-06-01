package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.KnnModel;
import Classification.Parameter.KnnParameter;
import Classification.Parameter.Parameter;

public class Knn extends Classifier {

    /**
     * Training algorithm for K-nearest neighbor classifier.
     * @param trainSet Training data given to the algorithm.
     * @param parameters K: k parameter of the K-nearest neighbor algorithm
     *                   distanceMetric: distance metric used to calculate the distance between two instances.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        model = new KnnModel(trainSet, ((KnnParameter) parameters).getK(), ((KnnParameter) parameters).getDistanceMetric());
    }

}
