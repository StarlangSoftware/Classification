package Classification.Attribute;

import java.io.Serializable;

public class DiscreteIndexedAttribute extends DiscreteAttribute implements Serializable {

    private int index;
    private int maxIndex;

    /**
     * Constructor for a discrete attribute.
     *
     * @param value Value of the attribute.
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

}
