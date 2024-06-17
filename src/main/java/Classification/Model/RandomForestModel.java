package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Parameter.Parameter;
import Classification.Parameter.RandomForestParameter;
import Sampling.Bootstrap;

import java.util.ArrayList;

public class RandomForestModel extends TreeEnsembleModel{

    @Override
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        int forestSize = ((RandomForestParameter) parameters).getEnsembleSize();
        forest = new ArrayList<>();
        for (int i = 0; i < forestSize; i++){
            Bootstrap<Instance> bootstrap = trainSet.bootstrap(i);
            DecisionTree tree = new DecisionTree(new DecisionNode(new InstanceList(bootstrap.getSample()), null, (RandomForestParameter) parameters, false));
            forest.add(tree);
        }
    }

}
