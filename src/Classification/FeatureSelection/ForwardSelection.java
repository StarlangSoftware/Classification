package Classification.FeatureSelection;

import java.util.ArrayList;

public class ForwardSelection extends SubSetSelection{

    public ForwardSelection() {
        super(new FeatureSubSet());
    }

    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        forward(result, current, numberOfFeatures);
        return result;
    }
}
