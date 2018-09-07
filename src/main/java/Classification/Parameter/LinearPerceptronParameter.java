package Classification.Parameter;

public class LinearPerceptronParameter extends Parameter{

    protected double learningRate;
    protected double etaDecrease;
    protected double crossValidationRatio;
    private int epoch;

    public LinearPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch){
        super(seed);
        this.learningRate = learningRate;
        this.etaDecrease = etaDecrease;
        this.crossValidationRatio = crossValidationRatio;
        this.epoch = epoch;
    }

    public double getLearningRate(){
        return learningRate;
    }

    public double getEtaDecrease(){
        return etaDecrease;
    }

    public double getCrossValidationRatio(){
        return crossValidationRatio;
    }

    public int getEpoch(){
        return epoch;
    }

}
