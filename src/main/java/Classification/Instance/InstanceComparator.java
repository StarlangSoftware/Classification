package Classification.Instance;

import Classification.Attribute.ContinuousAttribute;

import java.util.Comparator;

public class InstanceComparator implements Comparator<Instance> {

    private final int attributeIndex;

    /**
     * Constructor for instance comparator.
     *
     * @param attributeIndex Index of the attribute of which two instances will be compared.
     */
    public InstanceComparator(int attributeIndex) {
        this.attributeIndex = attributeIndex;
    }

    /**
     * Compares two instance on the values of the attribute with index attributeIndex.
     *
     * @param instance1 First instance to be compared
     * @param instance2 Second instance to be compared
     * @return -1 if the attribute value of the first instance is less than the attribute value of the second instance.
     * 1 if the attribute value of the first instance is greater than the attribute value of the second instance.
     * 0 if the attribute value of the first instance is equal to the attribute value of the second instance.
     */
    public int compare(Instance instance1, Instance instance2) {
        return Double.compare(((ContinuousAttribute) instance1.getAttribute(attributeIndex)).getValue(), ((ContinuousAttribute) instance2.getAttribute(attributeIndex)).getValue());
    }
}
