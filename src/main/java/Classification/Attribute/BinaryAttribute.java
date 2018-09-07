package Classification.Attribute;

import java.io.Serializable;

public class BinaryAttribute extends DiscreteAttribute implements Serializable{
    /**
     * Constructor for a binary discrete attribute. The attribute can take only two values "true" or "false".
     * @param value Value of the attribute. Can be true or false.
     */
    public BinaryAttribute(boolean value){
        super(Boolean.toString(value));
    }
}
