package Classification.Model.DecisionTree;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.ValidatedModel;

import java.io.Serializable;

public class DecisionTree extends ValidatedModel implements Serializable {

    private DecisionNode root;

    /**
     * The predict method  performs prediction on the root node of given instance, and if it is null, it returns the possible class labels.
     * Otherwise it returns the returned class labels.
     *
     * @param instance Instance make prediction.
     * @return Possible class labels.
     */
    public String predict(Instance instance) {
        String predictedClass = root.predict(instance);
        if ((predictedClass == null) && ((instance instanceof CompositeInstance))) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
        }
        return predictedClass;
    }

    /**
     * Constructor that sets root node of the decision tree.
     *
     * @param root DecisionNode type input.
     */
    public DecisionTree(DecisionNode root) {
        this.root = root;
    }

    /**
     * The prune method takes an {@link InstanceList} and  performs pruning to the root node.
     *
     * @param pruneSet {@link InstanceList} to perform pruning.
     */
    public void prune(InstanceList pruneSet) {
        root.prune(this, pruneSet);
    }
}
