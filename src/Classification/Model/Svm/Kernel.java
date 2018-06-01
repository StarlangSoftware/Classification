package Classification.Model.Svm;

import Classification.Parameter.SvmParameter;
import Util.Swap;

public class Kernel {
    private NodeList[] x;
    private double[] xSquare = null;
    private KernelType kernelType;
    private int degree;
    private double gamma;
    private double coefficient0;

    public Kernel(int l, NodeList[] x, KernelType kernelType, int degree, double gamma, double coefficient0){
        this.kernelType = kernelType;
        this.degree = degree;
        this.gamma = gamma;
        this.coefficient0 = coefficient0;
        this.x = new NodeList[l];
        for (int i = 0; i < l; i++){
            this.x[i] = x[i].clone();
        }
        if (kernelType.equals(KernelType.RBF)){
            xSquare = new double[l];
            for (int i = 0; i < l; i++){
                xSquare[i] = x[i].dot(x[i]);
            }
        }
    }

    public void swapIndex(int i, int j){
        NodeList tmp;
        tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
        if (xSquare != null){
            Swap.swap(xSquare, i, j);
        }
    }

    private double linear(int i, int j){
        return x[i].dot(x[j]);
    }

    private double polynom(int i, int j){
        return Math.pow(gamma * x[i].dot(x[j]) + coefficient0, degree);
    }

    private double rbf(int i, int j){
        return Math.exp(-gamma * (xSquare[i] + xSquare[j] - 2 * x[i].dot(x[j])));
    }

    private double sigmoid(int i, int j){
        return Math.tanh(gamma * x[i].dot(x[j]) + coefficient0);
    }

    public double function(int i, int j){
        switch (kernelType){
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

    public static double function(NodeList x, NodeList y, SvmParameter parameter){
        switch (parameter.getKernelType()){
            case LINEAR:
                return x.dot(y);
            case POLYNOM:
                return Math.pow(parameter.getGamma() * x.dot(y) + parameter.getCoefficient0(), parameter.getDegree());
            case RBF:
                double sum = 0;
                int px = 0, py = 0;
                while (px < x.size() && py < y.size()){
                    if (x.get(px).getIndex() == y.get(py).getIndex()){
                        double d = x.get(px).getValue() - y.get(py).getValue();
                        sum += d * d;
                        px++;
                        py++;
                    } else {
                        if (x.get(px).getIndex() > y.get(py).getIndex()){
                            sum += y.get(py).getValue() * y.get(py).getValue();
                            py++;
                        } else {
                            sum += x.get(px).getValue() * x.get(px).getValue();
                            px++;
                        }
                    }
                }
                while (px < x.size()){
                    sum += x.get(px).getValue() * x.get(px).getValue();
                    px++;
                }
                while (py < y.size()){
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
