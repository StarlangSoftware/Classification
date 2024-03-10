package Classification.Model.Svm;

import Classification.Parameter.SvmParameter;
import Util.Swap;

public class Kernel {
    private final NodeList[] x;
    private double[] xSquare = null;
    private final KernelType kernelType;
    private final int degree;
    private final double gamma;
    private final double coefficient0;

    /**
     * Constructor that sets x, kernelType, degree, gamma,xSquare, and coefficient0 with given inputs.
     *
     * @param l            Integer input.
     * @param x            NodeList Array.
     * @param kernelType   Kernel types; LINEAR, POLYNOM, RBF, SIGMOID.
     * @param degree       Degree of a node.
     * @param gamma        Gamma value.
     * @param coefficient0 First coefficient.
     */
    public Kernel(int l, NodeList[] x, KernelType kernelType, int degree, double gamma, double coefficient0) {
        this.kernelType = kernelType;
        this.degree = degree;
        this.gamma = gamma;
        this.coefficient0 = coefficient0;
        this.x = new NodeList[l];
        for (int i = 0; i < l; i++) {
            this.x[i] = x[i].clone();
        }
        if (kernelType.equals(KernelType.RBF)) {
            xSquare = new double[l];
            for (int i = 0; i < l; i++) {
                xSquare[i] = x[i].dot(x[i]);
            }
        }
    }

    /**
     * Swaps the kernel values of the instance i with instance j. Swaps also the square values.
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     */
    public void swapIndex(int i, int j) {
        NodeList tmp;
        tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
        if (xSquare != null) {
            Swap.swap(xSquare, i, j);
        }
    }

    /**
     * Calculates linear kernel value for two instances using dot product.
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     * @return Linear kernel value for two instances.
     */
    private double linear(int i, int j) {
        return x[i].dot(x[j]);
    }

    /**
     * Calculates polynomial kernel value for two instances. (gamma \sum_i x1_ix2_i)^d + x_0)
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     * @return Polynomial kernel value for two instances..
     */
    private double polynom(int i, int j) {
        return Math.pow(gamma * x[i].dot(x[j]) + coefficient0, degree);
    }

    /**
     * Calculates radial basis kernel value for two instances. e^(-gamma * (x1^2 + x2^2 - 2\sum_i x1_ix2_i)).
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     * @return Radial basis kernel value for two instances.
     */
    private double rbf(int i, int j) {
        return Math.exp(-gamma * (xSquare[i] + xSquare[j] - 2 * x[i].dot(x[j])));
    }

    /**
     * Calculates sigmoid kernel value for two instances. tanh(gamma * \sum_i x1_ix2_i + x_0).
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     * @return Sigmoid kernel value for two instances.
     */
    private double sigmoid(int i, int j) {
        return Math.tanh(gamma * x[i].dot(x[j]) + coefficient0);
    }

    /**
     * Calculates kernel value for two instances according to the kernel function. Calls respective kernel
     * functions such as linear kernel, polynomial kernel, rbf kernel, etc.
     *
     * @param i Index of the first instance.
     * @param j Index of the second instance.
     * @return Kernel value for two instances.
     */
    public double function(int i, int j) {
        switch (kernelType) {
            case LINEAR:
                return linear(i, j);
            case POLYNOM:
                return polynom(i, j);
            case RBF:
                return rbf(i, j);
            case SIGMOID:
                return sigmoid(i, j);
        }
        return 0;
    }

    /**
     * Calculates kernel value for two instances x and y according to the kernel. This function is different
     * from kernel_function in the way that kernel_function calculates kernel value for two instances of the data
     * array by specifying the indexes of those two instances. On the other hand, k_function calculates kernel value
     * for two usual instances.
     *
     * @param x         First instance.
     * @param y         Second instance.
     * @param parameter Kernel parameters including kernel type, gamma, x_0 etc.
     * @return Kernel value for two instances.
     */
    public static double function(NodeList x, NodeList y, SvmParameter parameter) {
        switch (parameter.getKernelType()) {
            case LINEAR:
                return x.dot(y);
            case POLYNOM:
                return Math.pow(parameter.getGamma() * x.dot(y) + parameter.getCoefficient0(), parameter.getDegree());
            case RBF:
                double sum = 0;
                int px = 0, py = 0;
                while (px < x.size() && py < y.size()) {
                    if (x.get(px).getIndex() == y.get(py).getIndex()) {
                        double d = x.get(px).getValue() - y.get(py).getValue();
                        sum += d * d;
                        px++;
                        py++;
                    } else {
                        if (x.get(px).getIndex() > y.get(py).getIndex()) {
                            sum += y.get(py).getValue() * y.get(py).getValue();
                            py++;
                        } else {
                            sum += x.get(px).getValue() * x.get(px).getValue();
                            px++;
                        }
                    }
                }
                while (px < x.size()) {
                    sum += x.get(px).getValue() * x.get(px).getValue();
                    px++;
                }
                while (py < y.size()) {
                    sum += y.get(py).getValue() * y.get(py).getValue();
                    py++;
                }
                return Math.exp(-parameter.getGamma() * sum);
            case SIGMOID:
                return Math.tanh(parameter.getGamma() * x.dot(y) + parameter.getCoefficient0());
        }
        return 0;
    }

}