package Classification.Model.Svm;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

public class Problem {
    private final int l;
    private final double[] y;
    private final NodeList[] x;

    /**
     * A constructor that sets x, y and l with given inputs.
     *
     * @param x NodeList array.
     * @param y Double array.
     */
    public Problem(NodeList[] x, double[] y) {
        this.x = x;
        this.y = y;
        this.l = y.length;
    }

    /**
     * A constructor that takes an {@link InstanceList} as an input and sets l, x, and y from that InstanceList.
     *
     * @param instanceList InstanceList to use.
     */
    public Problem(InstanceList instanceList) {
        Instance instance;
        l = instanceList.size();
        x = new NodeList[l];
        y = new double[l];
        for (int i = 0; i < instanceList.size(); i++) {
            instance = instanceList.get(i);
            x[i] = instance.toNodeList();
        }
    }

    /**
     * Accessor for l.
     *
     * @return L.
     */
    public int getL() {
        return l;
    }

    /**
     * Accessor for NodeList array x.
     *
     * @return x.
     */
    public NodeList[] getX() {
        return x;
    }

    /**
     * Accessor for double array y.
     *
     * @return y.
     */
    public double[] getY() {
        return y;
    }

}
