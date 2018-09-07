package Classification.Model;

import Classification.Instance.Instance;
import Classification.Model.DecisionTree.DecisionTree;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;

public class TreeEnsembleModel extends Model implements Serializable{

    private ArrayList<DecisionTree> forest;

    public String predict(Instance instance) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (DecisionTree tree:forest){
            distribution.addItem(tree.predict(instance));
        }
        return distribution.getMaxItem();
    }

    public TreeEnsembleModel(ArrayList<DecisionTree> forest){
        this.forest = forest;
    }

}
