package Classification.Parameter;

import Classification.Model.Svm.KernelType;

public class SvmParameter extends Parameter{
    private KernelType kernelType;
    private int degree;
    private double gamma;
    private double coefficient0;
    private double C;
    private boolean shrinking = true;

    public SvmParameter(int seed, KernelType kernelType, int degree, double gamma, double coefficient0, double C) {
        super(seed);
        this.kernelType = kernelType;
        this.degree = degree;
        this.gamma = gamma;
        this.coefficient0 = coefficient0;
        this.C = C;
    }

    public KernelType getKernelType() {
        return kernelType;
    }

    public int getDegree() {
        return degree;
    }

    public double getGamma() {
        return gamma;
    }

    public double getCoefficient0() {
        return coefficient0;
    }

    public double getC() {
        return C;
    }

    public boolean isShrinking() {
        return shrinking;
    }

}
