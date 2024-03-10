package Classification.Parameter;

public class MultiLayerPerceptronParameter extends LinearPerceptronParameter {

    private final int hiddenNodes;
    private final ActivationFunction activationFunction;

    /**
     * Parameters of the multi layer perceptron algorithm.
     *
     * @param seed                 Seed is used for random number generation.
     * @param learningRate         Double value for learning rate of the algorithm.
     * @param etaDecrease          Double value for decrease in eta of the algorithm.
     * @param crossValidationRatio Double value for cross validation ratio of the algorithm.
     * @param epoch                Integer value for epoch number of the algorithm.
     * @param hiddenNodes          Integer value for the number of hidden nodes.
     * @param activationFunction   Activation function
     */
    public MultiLayerPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, int hiddenNodes, ActivationFunction activationFunction) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.hiddenNodes = hiddenNodes;
        this.activationFunction = activationFunction;
    }

    /**
     * Accessor for the hiddenNodes.
     *
     * @return The hiddenNodes.
     */
    public int getHiddenNodes() {
        return hiddenNodes;
    }

    /**
     * Accessor for the activation function.
     *
     * @return The activation function.
     */
    public ActivationFunction getActivationFunction(){
        return activationFunction;
    }
}