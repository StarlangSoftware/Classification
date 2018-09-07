package Classification.FeatureSelection;

import java.util.ArrayList;

public class BackwardSelection extends SubSetSelection{

    public BackwardSelection(int numberOfFeatures) {
        super(new FeatureSubSet(numberOfFeatures));
    }

    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        backward(result, current);
        return result;
    }
}
