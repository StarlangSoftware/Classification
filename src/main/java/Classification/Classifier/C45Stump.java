package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Parameter.Parameter;

public class C45Stump extends Classifier {

    /**
     * Training algorithm for C4.5 Stump univariate decision tree classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        model = new DecisionTree(new DecisionNode(trainSet, null, null, true));
    }

    @Override
    public void loadModel(String fileName) {
        model = new DecisionTree(fileName);
    }

}
