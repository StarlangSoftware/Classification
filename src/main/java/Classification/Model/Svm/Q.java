package Classification.Model.Svm;

import Classification.Parameter.SvmParameter;
import Util.Swap;

public class Q {
    private Kernel kernel;
    private double[] y;

    /**
     * Constructor that sets problem, parameter and double array y.
     *
     * @param problem   {@link Problem} input.
     * @param parameter {@link SvmParameter} input.
     * @param y         A double array input.
     */
    public Q(Problem problem, SvmParameter parameter, double[] y) {
        kernel = new Kernel(problem.getL(), problem.getX(), parameter.getKernelType(), parameter.getDegree(), parameter.getGamma(), parameter.getCoefficient0());
        this.y = new double[problem.getL()];
        for (int i = 0; i < problem.getL(); i++) {
            this.y[i] = y[i];
        }
    }

    /**
     * The getQ method calculates y[i] * y[j] * kernel.function(i, j) and returns a new array with new values.
     *
     * @param i      Index to multiply values.
     * @param length Length of array.
     * @return A new array with new calculated values.
     */
    public double[] getQ(int i, int length) {
        double[] data;
        data = new double[length];
        for (int j = 0; j < length; j++) {
            data[j] = y[i] * y[j] * kernel.function(i, j);
        }
        return data;
    }

    /**
     * The swapIndex method swaps the ith element with jth element of array y.
     *
     * @param i First index to swap.
     * @param j Second index to swap.
     */
    public void swapIndex(int i, int j) {
        kernel.swapIndex(i, j);
        Swap.swap(y, i, j);
    }
}
