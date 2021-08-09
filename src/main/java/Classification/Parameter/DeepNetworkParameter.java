package Classification.Parameter;

import java.util.ArrayList;

public class DeepNetworkParameter extends LinearPerceptronParameter {
    private ArrayList<Integer> hiddenLayers;
    private ActivationFunction activationFunction;

    /**
     * Parameters of the deep network classifier.
     *
     * @param seed                 Seed is used for random number generation.
     * @param learningRate         Double value for learning rate of the algorithm.
     * @param etaDecrease          Double value for decrease in eta of the algorithm.
     * @param crossValidationRatio Double value for cross validation ratio of the algorithm.
     * @param epoch                Integer value for epoch number of the algorithm.
     * @param hiddenLayers         An integer {@link ArrayList} for hidden layers of the algorithm.
     */
    public DeepNetworkParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, ArrayList<Integer> hiddenLayers, ActivationFunction activationFunction) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.hiddenLayers = hiddenLayers;
        this.activationFunction = activationFunction;
    }

    /**
     * The layerSize method returns the size of the hiddenLayers {@link ArrayList}.
     *
     * @return The size of the hiddenLayers {@link ArrayList}.
     */
    public int layerSize() {
        return hiddenLayers.size();
    }

    /**
     * The getHiddenNodes method takes a layer index as an input and returns the element at the given index of hiddenLayers
     * {@link ArrayList}.
     *
     * @param layerIndex Index of the layer.
     * @return The element at the layerIndex of hiddenLayers {@link ArrayList}.
     */
    public int getHiddenNodes(int layerIndex) {
        return hiddenLayers.get(layerIndex);
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
