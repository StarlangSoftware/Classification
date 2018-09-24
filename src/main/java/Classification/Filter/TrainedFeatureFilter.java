package Classification.Filter;

import Classification.DataSet.DataSet;

public abstract class TrainedFeatureFilter extends FeatureFilter {

    protected abstract void train();

    /**
     * Constructor that sets the dataSet.
     *
     * @param dataSet DataSet that will bu used.
     */
    public TrainedFeatureFilter(DataSet dataSet) {
        super(dataSet);
    }
}
