package Classification.Parameter;

public class MultiLayerPerceptronParameter extends LinearPerceptronParameter{

    private int hiddenNodes;

    public MultiLayerPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, int hiddenNodes){
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.hiddenNodes = hiddenNodes;
    }

    public int getHiddenNodes(){
        return hiddenNodes;
    }
}
