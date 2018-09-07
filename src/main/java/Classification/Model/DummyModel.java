package Classification.Model;

import Classification.Classifier.Classifier;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;

public class DummyModel extends Model implements Serializable{

    private DiscreteDistribution distribution;

    public DummyModel(InstanceList trainSet) {
        this.distribution = trainSet.classDistribution();
    }

    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance)instance).getPossibleClassLabels();
            return distribution.getMaxItem(possibleClassLabels);
        } else {
            return distribution.getMaxItem();
        }
    }

}
