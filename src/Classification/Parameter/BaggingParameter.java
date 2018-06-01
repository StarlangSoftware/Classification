package Classification.Parameter;

public class BaggingParameter extends Parameter{

    protected int ensembleSize;

    public BaggingParameter(int seed, int ensembleSize){
        super(seed);
        this.ensembleSize = ensembleSize;
    }

    public int getEnsembleSize(){
        return ensembleSize;
    }

}
