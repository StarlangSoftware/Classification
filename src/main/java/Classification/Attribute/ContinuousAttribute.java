package Classification.Attribute;

import java.io.Serializable;
import java.util.ArrayList;

public class ContinuousAttribute extends Attribute implements Serializable {

    private double value;

    /**
     * Constructor for a continuous attribute.
     *
     * @param value Value of the attribute.
     */
    public ContinuousAttribute(double value) {
        this.value = value;
    }

    /**
     * Accessor method for value.
     *
     * @return value
     */
    public double getValue() {
        return value;
    }

    /**
     * Mutator method for value
     *
     * @param value New value of value.
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Converts value to {@link String}.
     *
     * @return String representation of value.
     */
    public String toString() {
        return "" + String.format("%.4f", value);
    }

    @Override
    public int continuousAttributeSize() {
        return 1;
    }

    @Override
    public ArrayList<Double> continuousAttributes() {
        ArrayList<Double> result = new ArrayList<>();
        result.add(value);
        return result;
    }
}
