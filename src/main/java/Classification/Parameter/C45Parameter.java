package Classification.Parameter;

public class C45Parameter extends Parameter{
    private boolean prune;
    private double crossValidationRatio;

    public C45Parameter(int seed, boolean prune, double crossValidationRatio) {
        super(seed);
        this.prune = prune;
        this.crossValidationRatio = crossValidationRatio;
    }

    public boolean isPrune(){
        return prune;
    }

    public double getCrossValidationRatio(){
        return crossValidationRatio;
    }
}
