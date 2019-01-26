package Classification.Attribute;

import java.io.Serializable;
import java.util.ArrayList;

public abstract class Attribute implements Serializable {
    public abstract int continuousAttributeSize();
    public abstract ArrayList<Double> continuousAttributes();
}
