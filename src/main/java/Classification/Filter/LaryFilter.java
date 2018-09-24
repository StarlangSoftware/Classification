package Classification.Filter;

import java.util.ArrayList;

import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;
import Math.DiscreteDistribution;

public abstract class LaryFilter extends FeatureFilter {
    protected ArrayList<DiscreteDistribution> attributeDistributions;

    /**
     * Constructor that sets the dataSet and all the attributes distributions.
     *
     * @param dataSet DataSet that will bu used.
     */
    public LaryFilter(DataSet dataSet) {
        super(dataSet);
        attributeDistributions = dataSet.getInstanceList().allAttributesDistribution();
    }

    /**
     * The removeDiscreteAttributes method takes an {@link Instance} as an input, and removes the discrete attributes from
     * given instance.
     *
     * @param instance Instance to removes attributes from.
     * @param size     Size of the given instance.
     */
    protected void removeDiscreteAttributes(Instance instance, int size) {
        int k = 0;
        for (int i = 0; i < size; i++) {
            if (attributeDistributions.get(i).size() > 0) {
                instance.removeAttribute(k);
            } else {
                k++;
            }
        }
    }

    /**
     * The removeDiscreteAttributes method removes the discrete attributes from dataDefinition.
     *
     * @param size Size of item that attributes will be removed.
     */
    protected void removeDiscreteAttributes(int size) {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int k = 0;
        for (int i = 0; i < size; i++) {
            if (attributeDistributions.get(i).size() > 0) {
                dataDefinition.removeAttribute(k);
            } else {
                k++;
            }
        }
    }

}
