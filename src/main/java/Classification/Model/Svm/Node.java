package Classification.Model.Svm;

public class Node {
    private int index;
    private double value;

    /**
     * Constructor that sets the value of index and value.
     *
     * @param index Index of the node.
     * @param value Value of the node.
     */
    public Node(int index, double value) {
        this.index = index;
        this.value = value;
    }

    /**
     * The clone method creates a new Node as a clone.
     *
     * @return New {@link Node}.
     */
    public Node clone() {
        return new Node(index, value);
    }

    /**
     * The getIndex method returns the index of a Node.
     *
     * @return The index of a Node.
     */
    public int getIndex() {
        return index;
    }

    /**
     * The getValue method returns the value of a Node.
     *
     * @return The value of a Node.
     */
    public double getValue() {
        return value;
    }

}
