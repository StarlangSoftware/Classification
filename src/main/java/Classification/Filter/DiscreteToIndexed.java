package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

public class DiscreteToIndexed extends LaryFilter {

    /**
     * Constructor for discrete to indexed filter.
     *
     * @param dataSet The dataSet whose instances whose discrete attributes will be converted to indexed attributes
     */
    public DiscreteToIndexed(DataSet dataSet) {
        super(dataSet);
    }

    /**
     * Converts discrete attributes of a single instance to indexed version.
     *
     * @param instance The instance to be converted.
     */
    protected void convertInstance(Instance instance) {
        int size = instance.attributeSize();
        for (int i = 0; i < size; i++) {
            if (attributeDistributions.get(i).size() > 0) {
                int index = attributeDistributions.get(i).getIndex(instance.getAttribute(i).toString());
                instance.addAttribute(new DiscreteIndexedAttribute(instance.getAttribute(i).toString(), index, attributeDistributions.get(i).size()));
            }
        }
        removeDiscreteAttributes(instance, size);
    }

    /**
     * Converts the data definition with discrete attributes, to data definition with DISCRETE_INDEXED attributes.
     */
    protected void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int size = dataDefinition.attributeCount();
        for (int i = 0; i < size; i++) {
            if (attributeDistributions.get(i).size() > 0) {
                dataDefinition.addAttribute(AttributeType.DISCRETE_INDEXED);
            }
        }
        removeDiscreteAttributes(size);
    }
}
