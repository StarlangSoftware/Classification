package Classification.Model.DecisionTree;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.ValidatedModel;
import Classification.Performance.ClassificationPerformance;

import java.io.Serializable;

public class DecisionTree extends ValidatedModel implements Serializable {

    private DecisionNode root;

    /**
     * Constructor that sets root node of the decision tree.
     *
     * @param root DecisionNode type input.
     */
    public DecisionTree(DecisionNode root) {
        this.root = root;
    }

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
     * The prune method takes a {@link DecisionNode} and an {@link InstanceList} as inputs. It checks the classification performance
     * of given InstanceList before pruning, i.e making a node leaf, and after pruning. If the after performance is better than the
     * before performance it prune the given InstanceList from the tree.
     *
     * @param node     DecisionNode that will be pruned if conditions hold.
     * @param pruneSet Small subset of tree that will be removed from tree.
     */
    public void pruneNode(DecisionNode node, InstanceList pruneSet) {
        ClassificationPerformance before, after;
        if (node.leaf)
            return;
        before = testClassifier(pruneSet);
        node.leaf = true;
        after = testClassifier(pruneSet);
        if (after.getAccuracy() < before.getAccuracy()) {
            node.leaf = false;
            for (DecisionNode child : node.children) {
                pruneNode(child, pruneSet);
            }
        }
    }

    /**
     * The prune method takes an {@link InstanceList} and  performs pruning to the root node.
     *
     * @param pruneSet {@link InstanceList} to perform pruning.
     */
    public void prune(InstanceList pruneSet) {
        pruneNode(root, pruneSet);
    }
}
