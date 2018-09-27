package Classification.InstanceList;

public class InstanceListOfSameClass extends InstanceList {

    private String classLabel;

    /**
     * Constructor for creating a new instance list with the same class labels.
     *
     * @param classLabel Class labels of instance list.
     */
    public InstanceListOfSameClass(String classLabel) {
        super();
        this.classLabel = classLabel;
    }

    /**
     * Accessor for the class labels.
     *
     * @return Class labels.
     */
    public String getClassLabel() {
        return classLabel;
    }
}

