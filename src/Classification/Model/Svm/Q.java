package Classification.Model.Svm;

import Classification.Parameter.SvmParameter;
import Util.Swap;

public class Q {
    private Kernel kernel;
    private double[] y;

    public Q(Problem problem, SvmParameter parameter, double[] y){
        kernel = new Kernel(problem.getL(), problem.getX(), parameter.getKernelType(), parameter.getDegree(), parameter.getGamma(), parameter.getCoefficient0());
        this.y = new double[problem.getL()];
        for (int i = 0; i < problem.getL(); i++){
            this.y[i] = y[i];
        }
    }

    public double[] getQ(int i, int length){
        double[] data;
        data = new double[length];
        for (int j = 0; j < length; j++){
            data[j] = y[i] * y[j] * kernel.function(i, j);
        }
        return data;
    }

    public void swapIndex(int i, int j){
        kernel.swapIndex(i, j);
        Swap.swap(y, i, j);
    }
}
