package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.DummyModel;
import Classification.Parameter.Parameter;

public class Dummy extends Classifier {

    /**
     * Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
     * the training data, there is no training.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        model = new DummyModel(trainSet);
    }

    @Override
    public void loadModel(String fileName) {
        model = new DummyModel(fileName);
    }
}
