package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.Svm.SvmModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.SvmParameter;

public class Svm extends Classifier {

    /**
     * Training algorithm for Support Vector Machine classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters Parameters of the SVM classifier algorithm.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        model = new SvmModel(trainSet, (SvmParameter) parameters);
    }
}
