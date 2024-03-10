package Classification.Classifier;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.TreeEnsembleModel;
import Classification.Parameter.BaggingParameter;
import Classification.Parameter.Parameter;
import Sampling.Bootstrap;

import java.util.ArrayList;

public class Bagging extends Classifier {

    /**
     * Bagging bootstrap ensemble method that creates individuals for its ensemble by training each classifier on a random
     * redistribution of the training set.
     * This training method is for a bagged decision tree classifier. 20 percent of the instances are left aside for pruning of the trees
     * 80 percent of the instances are used for training the trees. The number of trees (forestSize) is a parameter, and basically
     * the method will learn an ensemble of trees as a model.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the bagged forest.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        int forestSize = ((BaggingParameter) parameters).getEnsembleSize();
        ArrayList<DecisionTree> forest = new ArrayList<>();
        for (int i = 0; i < forestSize; i++) {
            Bootstrap<Instance> bootstrap = trainSet.bootstrap(i);
            DecisionTree tree = new DecisionTree(new DecisionNode(new InstanceList(bootstrap.getSample()), null, null, false));
            forest.add(tree);
        }
        model = new TreeEnsembleModel(forest);
    }

    @Override
    public void loadModel(String fileName) {
        model = new TreeEnsembleModel(fileName);
    }
}
