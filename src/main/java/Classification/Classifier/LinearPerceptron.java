package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.LinearPerceptronModel;
import Classification.Parameter.LinearPerceptronParameter;
import Classification.Parameter.Parameter;

import java.util.Random;

public class LinearPerceptron extends Classifier {

    /**
     * Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
     * data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
     * gradient descent.
     *
     * @param trainSet   Training data given to the algorithm
     * @param parameters Parameters of the linear perceptron.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = trainSet.stratifiedPartition(((LinearPerceptronParameter) parameters).getCrossValidationRatio(), new Random(parameters.getSeed()));
        model = new LinearPerceptronModel(partition.get(1), partition.get(0), (LinearPerceptronParameter) parameters);
    }
}
