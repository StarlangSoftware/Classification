package Classification.Model.DecisionTree;

import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;

public class DecisionStump extends DecisionTree {

    /**
     * Constructor that sets root node of the decision tree.
     *
     * @param root DecisionNode type input.
     */
    public DecisionStump(DecisionNode root) {
        super(root);
    }

    public DecisionStump(){

    }

    /**
     * Training algorithm for C4.5 Stump univariate decision tree classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        root = new DecisionNode(trainSet, null, null, true);
    }

}
