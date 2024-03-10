package Classification.Attribute;

import java.io.Serializable;
import java.util.ArrayList;

public class DiscreteAttribute extends Attribute implements Serializable{

    private final String value;

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
        if (value.equals(",")){
            return "comma";
        }
        return value;
    }

    @Override
    public int continuousAttributeSize() {
        return 0;
    }

    @Override
    public ArrayList<Double> continuousAttributes() {
        return new ArrayList<>();
    }
}
