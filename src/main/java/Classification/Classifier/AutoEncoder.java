package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.AutoEncoderModel;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;

import java.util.Random;

public class AutoEncoder extends Classifier{

    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))){
            throw new DiscreteFeaturesNotAllowed();
        }
        Partition partition = trainSet.stratifiedPartition(0.2, new Random(parameters.getSeed()));
        model = new AutoEncoderModel(partition.get(1), partition.get(0), (MultiLayerPerceptronParameter) parameters);
    }

    public Performance test(InstanceList testSet){
        return ((AutoEncoderModel) model).testAutoEncoder(testSet);
    }

}
