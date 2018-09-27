package Classification.Parameter;

public class MultiLayerPerceptronParameter extends LinearPerceptronParameter {

    private int hiddenNodes;

    /**
     * Parameters of the multi layer perceptron algorithm.
     *
     * @param seed                 Seed is used for random number generation.
     * @param learningRate         Double value for learning rate of the algorithm.
     * @param etaDecrease          Double value for decrease in eta of the algorithm.
     * @param crossValidationRatio Double value for cross validation ratio of the algorithm.
     * @param epoch                Integer value for epoch number of the algorithm.
     * @param hiddenNodes          Integer value for the number of hidden nodes.
     */
    public MultiLayerPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, int hiddenNodes) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.hiddenNodes = hiddenNodes;
    }

    /**
     * Accessor for the hiddenNodes.
     *
     * @return The hiddenNodes.
     */
    public int getHiddenNodes() {
        return hiddenNodes;
    }
}