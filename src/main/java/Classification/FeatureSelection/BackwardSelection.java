package Classification.FeatureSelection;

import java.util.ArrayList;

public class BackwardSelection extends SubSetSelection {

    /**
     * Constructor that creates a new {@link FeatureSubSet} and initializes indexList with given number of features.
     *
     * @param numberOfFeatures Indicates the indices of indexList.
     */
    public BackwardSelection(int numberOfFeatures) {
        super(new FeatureSubSet(numberOfFeatures));
    }

    /**
     * The operator method calls backward method which starts with all the features and removes the least significant feature at each iteration.
     *
     * @param current          FeatureSubset that will be added to new ArrayList.
     * @param numberOfFeatures Indicates the indices of indexList.
     * @return ArrayList of FeatureSubSets created from backward.
     */
    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        backward(result, current);
        return result;
    }
}
