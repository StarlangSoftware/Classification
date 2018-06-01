package Classification.Parameter;

import java.util.ArrayList;

public class DeepNetworkParameter extends LinearPerceptronParameter{
    private ArrayList<Integer> hiddenLayers;

    public DeepNetworkParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, ArrayList<Integer> hiddenLayers) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.hiddenLayers = hiddenLayers;
    }

    public int layerSize(){
        return hiddenLayers.size();
    }

    public int getHiddenNodes(int layerIndex){
        return hiddenLayers.get(layerIndex);
    }
}
