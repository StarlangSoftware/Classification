package Classification.Model;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;

public class DummyModel extends Model implements Serializable {

    private DiscreteDistribution distribution;

    /**
     * Constructor which sets the distribution using the given {@link InstanceList}.
     *
     * @param trainSet {@link InstanceList} which is used to get the class distribution.
     */
    public DummyModel(InstanceList trainSet) {
        this.distribution = trainSet.classDistribution();
    }

    /**
     * The predict method takes an Instance as an input and returns the entry of distribution which has the maximum value.
     *
     * @param instance Instance to make prediction.
     * @return The entry of distribution which has the maximum value.
     */
    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
            return distribution.getMaxItem(possibleClassLabels);
        } else {
            return distribution.getMaxItem();
        }
    }

}
