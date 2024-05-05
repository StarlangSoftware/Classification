package Classification.Classifier;

import Classification.Instance.Instance;
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
     *
     * @param trainSet   Training data given to the algorithm
     * @param parameters Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the random forest.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        int forestSize = ((RandomForestParameter) parameters).getEnsembleSize();
        ArrayList<DecisionTree> forest = new ArrayList<>();
        for (int i = 0; i < forestSize; i++){
            Bootstrap<Instance> bootstrap = trainSet.bootstrap(i);
            DecisionTree tree = new DecisionTree(new DecisionNode(new InstanceList(bootstrap.getSample()), null, (RandomForestParameter) parameters, false));
            forest.add(tree);
        }
        model = new TreeEnsembleModel(forest);
    }

    /**
     * Loads the random forest model from an input file.
     * @param fileName File name of the random forest model.
     */
    @Override
    public void loadModel(String fileName) {
        model = new TreeEnsembleModel(fileName);
    }
}
