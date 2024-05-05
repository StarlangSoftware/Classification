package Classification.Attribute;

import java.io.Serializable;
import java.util.ArrayList;

public class DiscreteIndexedAttribute extends DiscreteAttribute implements Serializable {

    private final int index;
    private final int maxIndex;

    /**
     * Constructor for a discrete attribute.
     *
     * @param value Value of the attribute.
     * @param index Index of the attribute.
     * @param maxIndex Maximum index of the attribute.
     */
    public DiscreteIndexedAttribute(String value, int index, int maxIndex) {
        super(value);
        this.index = index;
        this.maxIndex = maxIndex;
    }

    /**
     * Accessor method for index.
     *
     * @return index.
     */
    public int getIndex() {
        return index;
    }

    /**
     * Accessor method for maxIndex.
     *
     * @return maxIndex.
     */
    public int getMaxIndex() {
        return maxIndex;
    }

    @Override
    public int continuousAttributeSize() {
        return maxIndex;
    }

    /**
     * Returns 1 of L encoding of this attribute. Only the value at index maxIndex is 1, others are 0.
     * @return 1 of L encoding of this attribute.
     */
    @Override
    public ArrayList<Double> continuousAttributes() {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < maxIndex; i++) {
            if (i != index) {
                result.add(0.0);
            } else {
                result.add(1.0);
            }
        }
        return result;
    }

}
