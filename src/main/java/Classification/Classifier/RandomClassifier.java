package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.RandomModel;
import Classification.Parameter.Parameter;

import java.util.ArrayList;

public class RandomClassifier extends Classifier {

    /**
     * Training algorithm for random classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    @Override
    public void train(InstanceList trainSet, Parameter parameters) {
        model = new RandomModel(new ArrayList<String>(trainSet.classDistribution().keySet()));
    }

}
