package Classification.Performance;

public class DetailedClassificationPerformance extends ClassificationPerformance {
    private ConfusionMatrix confusionMatrix;

    /**
     * A constructor that  sets the accuracy and errorRate as 1 - accuracy via given {@link ConfusionMatrix} and also sets the confusionMatrix.
     *
     * @param confusionMatrix {@link ConfusionMatrix} input.
     */
    public DetailedClassificationPerformance(ConfusionMatrix confusionMatrix) {
        super(confusionMatrix.getAccuracy());
        this.confusionMatrix = confusionMatrix;
    }

    /**
     * Accessor for the confusionMatrix.
     *
     * @return ConfusionMatrix.
     */
    public ConfusionMatrix getConfusionMatrix() {
        return confusionMatrix;
    }

}
