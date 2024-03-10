package Classification.InstanceList;

public class InstanceListOfSameClass extends InstanceList {

    private final String classLabel;

    /**
     * Constructor for creating a new instance list with the same class label.
     *
     * @param classLabel Class label of instance list.
     */
    public InstanceListOfSameClass(String classLabel) {
        super();
        this.classLabel = classLabel;
    }

    /**
     * Accessor for the class label.
     *
     * @return Class label.
     */
    public String getClassLabel() {
        return classLabel;
    }
}

