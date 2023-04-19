package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.RandomModel;
import Classification.Parameter.C45Parameter;
import Classification.Parameter.Parameter;
import Classification.InstanceList.Partition;

import java.util.Random;

public class C45 extends Classifier {

    /**
     * Training algorithm for C4.5 univariate decision tree classifier. 20 percent of the data are left aside for pruning
     * 80 percent of the data is used for constructing the tree.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        DecisionTree tree;
        if (((C45Parameter) parameters).isPrune()) {
            Partition partition = new Partition(trainSet, ((C45Parameter) parameters).getCrossValidationRatio(), new Random(parameters.getSeed()), true);
            tree = new DecisionTree(new DecisionNode(partition.get(1), null, null, false));
            tree.prune(partition.get(0));
        } else {
            tree = new DecisionTree(new DecisionNode(trainSet, null, null, false));
        }
        model = tree;
    }

    @Override
    public void loadModel(String fileName) {
        model = new DecisionTree(fileName);
    }
}
