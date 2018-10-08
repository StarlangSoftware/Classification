package Classification.Performance;

public class Performance {
    protected double errorRate;

    /**
     * Constructor that sets the error rate.
     *
     * @param errorRate Double input.
     */
    public Performance(double errorRate) {
        this.errorRate = errorRate;
    }

    /**
     * Accessor for the error rate.
     *
     * @return Double errorRate.
     */
    public double getErrorRate() {
        return errorRate;
    }
}
