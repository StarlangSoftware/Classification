package Classification.Instance;

import java.util.Comparator;

public class InstanceClassComparator implements Comparator<Instance> {

    /**
     * Compares two {@link Instance} inputs and returns a positive value if the first input's class label is greater
     * than the second's class label input lexicographically.
     *
     * @param o1 First {@link Instance} to be compared.
     * @param o2 Second {@link Instance} to be compared.
     * @return Negative value if the class label of the first instance is less than the class label of the second instance.
     * Positive value if the class label of the first instance is greater than the class label of the second instance.
     * 0 if the class label of the first instance is equal to the class label of the second instance.
     */
    public int compare(Instance o1, Instance o2) {
        return o1.getClassLabel().compareTo(o2.getClassLabel());
    }
}
