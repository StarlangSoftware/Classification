package Classification.Attribute;

import java.io.Serializable;

public class DiscreteAttribute extends Attribute implements Serializable{

    private String value = "NULL";

    /**
     * Constructor for a discrete attribute.
     *
     * @param value Value of the attribute.
     */
    public DiscreteAttribute(String value) {
        this.value = value;
    }

    /**
     * Accessor method for value.
     *
     * @return value
     */
    public String getValue() {
        return value;
    }

    /**
     * Converts value to {@link String}.
     *
     * @return String representation of value.
     */
    public String toString() {
        return value;
    }
}
