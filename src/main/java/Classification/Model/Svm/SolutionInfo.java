package Classification.Model.Svm;

public class SolutionInfo {
    private final double rho;
    private final double[] alpha;

    /**
     * Constructor that sets rho and alpha values.
     *
     * @param rho   Double input rho.
     * @param alpha Double array of alpha values.
     */
    public SolutionInfo(double rho, double[] alpha) {
        this.rho = rho;
        this.alpha = alpha;
    }

    /**
     * Accessor for rho value.
     *
     * @return Rho value.
     */
    public double getRho() {
        return rho;
    }

    /**
     * Accessor for alpha value. Returns the alpha at given index.
     *
     * @param index Index to retrieve alpha.
     * @return The alpha at given index.
     */
    public double getAlpha(int index) {
        return alpha[index];
    }

    /**
     * Mutator for alpha values.
     *
     * @param index Index to change alpha value.
     * @param value Value to be set as new alpha value at given index.
     */
    public void setAlpha(int index, double value) {
        alpha[index] = value;
    }

}
