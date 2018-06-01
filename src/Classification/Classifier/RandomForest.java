package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.TreeEnsembleModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.RandomForestParameter;
import Sampling.Bootstrap;

import java.util.ArrayList;

public class RandomForest extends Classifier{

    /**
     * Training algorithm for random forest classifier. Basically the algorithm creates K distinct decision trees from
     * K bootstrap samples of the original training set.
     * @param trainSet Training data given to the algorithm
     * @param parameters Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the random forest.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        int forestSize = ((RandomForestParameter) parameters).getEnsembleSize();
        ArrayList<DecisionTree> forest = new ArrayList<DecisionTree>();
        for (int i = 0; i < forestSize; i++){
            Bootstrap bootstrap = trainSet.bootstrap(i);
            forest.add(new DecisionTree(new DecisionNode(new InstanceList(bootstrap.getSample()), null, (RandomForestParameter) parameters, false)));
        }
        model = new TreeEnsembleModel(forest);
    }
}
