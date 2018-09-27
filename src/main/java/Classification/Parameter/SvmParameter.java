package Classification.Parameter;

import Classification.Model.Svm.KernelType;

public class SvmParameter extends Parameter {
    private KernelType kernelType;
    private int degree;
    private double gamma;
    private double coefficient0;
    private double C;
    private boolean shrinking = true;

    /**
     * Parameters of the Support Vector Machine classifier.
     *
     * @param seed         Seed is used for random number generation.
     * @param kernelType   Specifies the {@link KernelType} to be used in the algorithm. It can be ona of the LINEAR, POLYNOM, RBF or SIGMOID.
     * @param degree       Degree of the polynomial kernel function.
     * @param gamma        Kernel coefficient for ?RBF, POLYNOM and SIGMOID.
     * @param coefficient0 Independent term in kernel function. It is only significant in POLYNOM and SIGMOID.
     * @param C            C is a regularization parameter which controls the trade off between the achieving a low
     *                     training error and a low testing error.
     */
    public SvmParameter(int seed, KernelType kernelType, int degree, double gamma, double coefficient0, double C) {
        super(seed);
        this.kernelType = kernelType;
        this.degree = degree;
        this.gamma = gamma;
        this.coefficient0 = coefficient0;
        this.C = C;
    }

    /**
     * Accessor for the kernelType.
     *
     * @return The kernelType.
     */
    public KernelType getKernelType() {
        return kernelType;
    }

    /**
     * Accessor for the degree.
     *
     * @return The degree.
     */
    public int getDegree() {
        return degree;
    }

    /**
     * Accessor for the gamma.
     *
     * @return The gamma.
     */
    public double getGamma() {
        return gamma;
    }

    /**
     * Accessor for the coefficient0.
     *
     * @return The coefficient0.
     */
    public double getCoefficient0() {
        return coefficient0;
    }

    /**
     * Accessor for the C.
     *
     * @return The C.
     */
    public double getC() {
        return C;
    }

    /**
     * Accessor for the shrinking.
     *
     * @return The shrinking.
     */
    public boolean isShrinking() {
        return shrinking;
    }

}