package Classification.Filter;

import Classification.Attribute.ContinuousAttribute;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

public class Normalize extends FeatureFilter {
    private Instance averageInstance, standardDeviationInstance;

    /**
     * Constructor for normalize feature filter. It calculates and stores the mean (m) and standard deviation (s) of
     * the sample.
     *
     * @param dataSet Instances whose continuous attribute values will be normalized.
     */
    public Normalize(DataSet dataSet) {
        super(dataSet);
        averageInstance = dataSet.getInstanceList().average();
        standardDeviationInstance = dataSet.getInstanceList().standardDeviation();
    }

    /**
     * Normalizes the continuous attributes of a single instance. For all i, new x_i = (x_i - m_i) / s_i.
     *
     * @param instance Instance whose attributes will be normalized.
     */
    protected void convertInstance(Instance instance) {
        for (int i = 0; i < instance.attributeSize(); i++) {
            if (instance.getAttribute(i) instanceof ContinuousAttribute) {
                ContinuousAttribute xi = (ContinuousAttribute) instance.getAttribute(i);
                ContinuousAttribute mi = (ContinuousAttribute) averageInstance.getAttribute(i);
                ContinuousAttribute si = (ContinuousAttribute) standardDeviationInstance.getAttribute(i);
                xi.setValue((xi.getValue() - mi.getValue()) / si.getValue());
            }
        }
    }

    protected void convertDataDefinition() {
    }
}
