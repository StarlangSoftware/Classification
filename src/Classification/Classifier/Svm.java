package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.Svm.SvmModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.SvmParameter;

public class Svm extends Classifier{

    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))){
            throw new DiscreteFeaturesNotAllowed();
        }
        model = new SvmModel(trainSet, (SvmParameter) parameters);
    }
}
