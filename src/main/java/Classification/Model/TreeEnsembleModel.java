package Classification.Model;

import Classification.Instance.Instance;
import Classification.Model.DecisionTree.DecisionTree;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;

public class TreeEnsembleModel extends Model implements Serializable {

    private ArrayList<DecisionTree> forest;

    /**
     * The predict method takes an {@link Instance} as an input and loops through the {@link ArrayList} of {@link DecisionTree}s.
     * Makes prediction for the items of that ArrayList and returns the maximum item of that ArrayList.
     *
     * @param instance Instance to make prediction.
     * @return The maximum prediction of a given Instance.
     */
    public String predict(Instance instance) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (DecisionTree tree : forest) {
            distribution.addItem(tree.predict(instance));
        }
        return distribution.getMaxItem();
    }

    /**
     * A constructor which sets the {@link ArrayList} of {@link DecisionTree} with given input.
     *
     * @param forest An {@link ArrayList} of {@link DecisionTree}.
     */
    public TreeEnsembleModel(ArrayList<DecisionTree> forest) {
        this.forest = forest;
    }

}
