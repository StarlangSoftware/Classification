package Classification.FeatureSelection;

import java.util.ArrayList;

public class ForwardSelection extends SubSetSelection{

    /**
     * Constructor that creates a new {@link FeatureSubSet}.
     */
    public ForwardSelection() {
        super(new FeatureSubSet());
    }

    /**
     * The operator method calls forward method which starts with having no feature in the model. In each iteration,
     * it keeps adding the features that are not currently listed.
     *
     * @param current          FeatureSubset that will be added to new ArrayList.
     * @param numberOfFeatures Indicates the indices of indexList.
     * @return ArrayList of FeatureSubSets created from forward.
     */
    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        forward(result, current, numberOfFeatures);
        return result;
    }
}
