package Classification.Parameter;

public class LinearPerceptronParameter extends Parameter {

    protected double learningRate;
    protected double etaDecrease;
    protected double crossValidationRatio;
    private int epoch;

    /**
     * Parameters of the linear perceptron algorithm.
     *
     * @param seed                 Seed is used for random number generation.
     * @param learningRate         Double value for learning rate of the algorithm.
     * @param etaDecrease          Double value for decrease in eta of the algorithm.
     * @param crossValidationRatio Double value for cross validation ratio of the algorithm.
     * @param epoch                Integer value for epoch number of the algorithm.
     */
    public LinearPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch) {
        super(seed);
        this.learningRate = learningRate;
        this.etaDecrease = etaDecrease;
        this.crossValidationRatio = crossValidationRatio;
        this.epoch = epoch;
    }

    /**
     * Accessor for the learningRate.
     *
     * @return The learningRate.
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Accessor for the etaDecrease.
     *
     * @return The etaDecrease.
     */
    public double getEtaDecrease() {
        return etaDecrease;
    }

    /**
     * Accessor for the crossValidationRatio.
     *
     * @return The crossValidationRatio.
     */
    public double getCrossValidationRatio() {
        return crossValidationRatio;
    }

    /**
     * Accessor for the epoch.
     *
     * @return The epoch.
     */
    public int getEpoch() {
        return epoch;
    }

}