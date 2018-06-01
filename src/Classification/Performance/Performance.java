package Classification.Performance;

public class Performance {
    protected double errorRate;

    public Performance(double errorRate){
        this.errorRate = errorRate;
    }

    public double getErrorRate(){
        return errorRate;
    }
}
