package Classification.Model.Svm;

import java.util.ArrayList;

public class NodeList {
    private ArrayList<Node> nodes;

    /**
     * Constructor that Creates an ArrayList of Nodes and initialize with the given values.
     *
     * @param values An ArrayList of values of nodes.
     */
    public NodeList(ArrayList<Double> values) {
        nodes = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            if (values.get(i) != 0) {
                nodes.add(new Node(i, values.get(i)));
            }
        }
    }

    /**
     * An empty constructor.
     */
    public NodeList() {
    }

    /**
     * The clone method creates a new {@link NodeList} and as the clone of the initial {@link NodeList}.
     *
     * @return A clone of the {@link NodeList}.
     */
    public NodeList clone() {
        NodeList result = new NodeList();
        result.nodes = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            result.nodes.add(nodes.get(i).clone());
        }
        return result;
    }

    /**
     * The dot method takes a {@link NodeList} as an input and returns the dot product of given {@link NodeList}
     * and initial NodeList.
     *
     * @param nodeList NodeList to find the dot product.
     * @return Dot product.
     */
    public double dot(NodeList nodeList) {
        double sum = 0;
        int px = 0, py = 0;
        while (px < nodes.size() && py < nodeList.nodes.size()) {
            if (nodes.get(px).getIndex() == nodeList.nodes.get(py).getIndex()) {
                sum += nodes.get(px).getValue() * nodeList.nodes.get(py).getValue();
                px++;
                py++;
            } else {
                if (nodes.get(px).getIndex() > nodeList.nodes.get(py).getIndex()) {
                    py++;
                } else {
                    px++;
                }
            }
        }
        return sum;
    }

    /**
     * The get method returns the Node at given index.
     *
     * @param index Index to find a Node of the NodeList.
     * @return The Node at given index.
     */
    public Node get(int index) {
        return nodes.get(index);
    }

    /**
     * The size method returns the size of the NodeList.
     *
     * @return The size of the NodeList.
     */
    public int size() {
        return nodes.size();
    }
}
