package Classification.Attribute;

import java.io.Serializable;

public class BinaryAttribute extends DiscreteAttribute implements Serializable {
    /**
     * Constructor for a binary discrete attribute and the attribute can take only two values "True" or "False" as an input.
     *
     * @param value Boolean value of the attribute.
     */
    public BinaryAttribute(boolean value) {
        super(Boolean.toString(value));
    }
}
