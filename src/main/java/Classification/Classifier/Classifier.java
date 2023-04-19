package Classification.Classifier;

import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.Performance.ConfusionMatrix;
import Classification.Performance.DetailedClassificationPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.Model;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;
import DataStructure.*;

import java.util.ArrayList;

public abstract class Classifier {

    protected Model model;

    public abstract void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed;

    public abstract void loadModel(String fileName);

    /**
     * Checks given instance's attribute and returns true if it is a discrete indexed attribute, false otherwise.
     *
     * @param instance Instance to check.
     * @return True if instance is a discrete indexed attribute, false otherwise.
     */
    public boolean discreteCheck(Instance instance) {
        for (int i = 0; i < instance.attributeSize(); i++) {
            if (instance.getAttribute(i) instanceof DiscreteAttribute && !(instance.getAttribute(i) instanceof DiscreteIndexedAttribute)) {
                return false;
            }
        }
        return true;
    }

    /**
     * TestClassification an instance list with the current model.
     *
     * @param testSet Test data (list of instances) to be tested.
     * @return The accuracy (and error) of the model as an instance of Performance class.
     */
    public Performance test(InstanceList testSet) {
        ArrayList<String> classLabels = testSet.getUnionOfPossibleClassLabels();
        ConfusionMatrix confusion = new ConfusionMatrix(classLabels);
        for (int i = 0; i < testSet.size(); i++) {
            Instance instance = testSet.get(i);
            confusion.classify(instance.getClassLabel(), model.predict(instance));
        }
        return new DetailedClassificationPerformance(confusion);
    }

    /**
     * Runs current classifier with the given train and test data.
     *
     * @param parameter Parameter of the classifier to be trained.
     * @param trainSet  Training data to be used in training the classifier.
     * @param testSet   Test data to be tested after training the model.
     * @return The accuracy (and error) of the trained model as an instance of Performance class.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public Performance singleRun(Parameter parameter, InstanceList trainSet, InstanceList testSet) throws DiscreteFeaturesNotAllowed {
        train(trainSet, parameter);
        return test(testSet);
    }

    /**
     * Accessor for the model.
     *
     * @return Model.
     */
    public Model getModel() {
        return model;
    }

}
