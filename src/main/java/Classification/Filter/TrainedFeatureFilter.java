package Classification.Filter;

import Classification.DataSet.DataSet;

public abstract class TrainedFeatureFilter extends FeatureFilter{

    protected abstract void train();

    public TrainedFeatureFilter(DataSet dataSet) {
        super(dataSet);
    }
}
