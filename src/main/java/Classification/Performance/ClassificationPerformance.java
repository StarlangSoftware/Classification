package Classification.Performance;

public class ClassificationPerformance extends Performance {

    private final double accuracy;

    /**
     * A constructor that sets the accuracy and errorRate as 1 - accuracy via given accuracy.
     *
     * @param accuracy Double value input.
     */
    public ClassificationPerformance(double accuracy) {
        super(1 - accuracy);
        this.accuracy = accuracy;
    }

    /**
     * A constructor that sets the accuracy and errorRate via given input.
     *
     * @param accuracy  Double value input.
     * @param errorRate Double value input.
     */
    public ClassificationPerformance(double accuracy, double errorRate) {
        super(errorRate);
        this.accuracy = accuracy;
    }

    /**
     * Accessor for the accuracy.
     *
     * @return Accuracy value.
     */
    public double getAccuracy() {
        return accuracy;
    }

}
