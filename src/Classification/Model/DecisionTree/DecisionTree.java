package Classification.Model.DecisionTree;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.ValidatedModel;

import java.io.Serializable;

public class DecisionTree extends ValidatedModel implements Serializable {

    private DecisionNode root;

    public String predict(Instance instance) {
        String predictedClass = root.predict(instance);
        if ((predictedClass == null) && ((instance instanceof CompositeInstance))) {
            predictedClass = ((CompositeInstance)instance).getPossibleClassLabels().get(0);
        }
        return predictedClass;
    }

    public DecisionTree(DecisionNode root){
        this.root = root;
    }

    public void prune(InstanceList pruneSet){
        root.prune(this, pruneSet);
    }
}
