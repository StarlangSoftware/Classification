package Classification.FeatureSelection;

import java.util.ArrayList;

public class FloatingSelection extends SubSetSelection {

    /**
     * Constructor that creates a new {@link FeatureSubSet}.
     */
    public FloatingSelection() {
        super(new FeatureSubSet());
    }

    /**
     * The operator method calls forward and backward methods.
     *
     * @param current          {@link FeatureSubSet} input.
     * @param numberOfFeatures Indicates the indices of indexList.
     * @return ArrayList of FeatureSubSet.
     */
    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        forward(result, current, numberOfFeatures);
        backward(result, current);
        return result;
    }

}
