package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.MultiLayerPerceptronModel;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Parameter.Parameter;

import java.util.Random;

public class MultiLayerPerceptron extends Classifier {

    /**
     * Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as cross-validation
     * data used for selecting the best weights. 80 percent of the data is used for training the multilayer perceptron with
     * gradient descent.
     *
     * @param trainSet   Training data given to the algorithm
     * @param parameters Parameters of the multilayer perceptron.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = new Partition(trainSet, ((MultiLayerPerceptronParameter) parameters).getCrossValidationRatio(), new Random(parameters.getSeed()), true);
        model = new MultiLayerPerceptronModel(partition.get(1), partition.get(0), (MultiLayerPerceptronParameter) parameters);
    }

    /**
     * Loads the multi-layer perceptron model from an input file.
     * @param fileName File name of the multi-layer perceptron model.
     */
    @Override
    public void loadModel(String fileName) {
        model = new MultiLayerPerceptronModel(fileName);
    }
}
