package Classification.FeatureSelection;

import java.util.ArrayList;

public class FloatingSelection extends SubSetSelection{

    public FloatingSelection() {
        super(new FeatureSubSet());
    }

    protected ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures) {
        ArrayList<FeatureSubSet> result = new ArrayList<>();
        forward(result, current, numberOfFeatures);
        backward(result, current);
        return result;
    }

}
