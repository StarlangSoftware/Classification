package Classification.Parameter;

public class C45Parameter extends Parameter {
    private final boolean prune;
    private final double crossValidationRatio;

    /**
     * Parameters of the C4.5 univariate decision tree classifier.
     *
     * @param seed                 Seed is used for random number generation.
     * @param prune                Boolean value for prune.
     * @param crossValidationRatio Double value for cross crossValidationRatio ratio.
     */
    public C45Parameter(int seed, boolean prune, double crossValidationRatio) {
        super(seed);
        this.prune = prune;
        this.crossValidationRatio = crossValidationRatio;
    }

    /**
     * Accessor for the prune.
     *
     * @return Prune.
     */
    public boolean isPrune() {
        return prune;
    }

    /**
     * Accessor for the crossValidationRatio.
     *
     * @return crossValidationRatio.
     */
    public double getCrossValidationRatio() {
        return crossValidationRatio;
    }
}
