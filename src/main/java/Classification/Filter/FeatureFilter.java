package Classification.Filter;

import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

import java.util.ArrayList;

public abstract class FeatureFilter {
    protected DataSet dataSet;

    protected abstract void convertInstance(Instance instance);

    protected abstract void convertDataDefinition();

    /**
     * Constructor that sets the dataSet.
     *
     * @param dataSet DataSet that will bu used.
     */
    public FeatureFilter(DataSet dataSet) {
        this.dataSet = dataSet;
    }

    /**
     * Feature converter for a list of instances. Using the abstract method convertInstance, each instance in the
     * instance list will be converted.
     */
    public void convert() {
        ArrayList<Instance> instances = dataSet.getInstances();
        for (Instance instance : instances) {
            convertInstance(instance);
        }
        convertDataDefinition();
    }
}
