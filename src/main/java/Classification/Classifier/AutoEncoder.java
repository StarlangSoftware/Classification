package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.AutoEncoderModel;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;

import java.util.Random;

public class AutoEncoder extends Classifier {

    /**
     * Training algorithm for auto encoders. An auto encoder is a neural network which attempts to replicate its input at its output.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters Parameters of the auto encoder.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = trainSet.stratifiedPartition(0.2, new Random(parameters.getSeed()));
        model = new AutoEncoderModel(partition.get(1), partition.get(0), (MultiLayerPerceptronParameter) parameters);
    }

    /**
     * A performance test for an auto encoder with the given test set..
     *
     * @param testSet Test data (list of instances) to be tested.
     * @return Error rate.
     */
    public Performance test(InstanceList testSet) {
        return ((AutoEncoderModel) model).testAutoEncoder(testSet);
    }

}
