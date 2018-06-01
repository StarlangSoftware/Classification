package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.DeepNetworkModel;
import Classification.Parameter.DeepNetworkParameter;
import Classification.Parameter.Parameter;

import java.util.Random;

public class DeepNetwork extends Classifier{

    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))){
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = trainSet.stratifiedPartition(((DeepNetworkParameter) parameters).getCrossValidationRatio(), new Random(parameters.getSeed()));
        model = new DeepNetworkModel(partition.get(1), partition.get(0), (DeepNetworkParameter) parameters);
    }

}
