package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.DeepNetworkModel;
import Classification.Parameter.DeepNetworkParameter;
import Classification.Parameter.Parameter;

import java.util.Random;

public class DeepNetwork extends Classifier {

    /**
     * Training algorithm for deep network classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = new Partition(trainSet, ((DeepNetworkParameter) parameters).getCrossValidationRatio(), new Random(parameters.getSeed()), true);
        model = new DeepNetworkModel(partition.get(1), partition.get(0), (DeepNetworkParameter) parameters);
    }

    /**
     * Loads the deep network model from an input file.
     * @param fileName File name of the deep network model.
     */
    @Override
    public void loadModel(String fileName) {
        model = new DeepNetworkModel(fileName);
    }

}
