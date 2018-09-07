package Classification.Performance;

public class ClassificationPerformance extends Performance{

    private double accuracy;

    public ClassificationPerformance(double accuracy){
        super(1 - accuracy);
        this.accuracy = accuracy;
    }

    public ClassificationPerformance(double accuracy, double errorRate){
        super(errorRate);
        this.accuracy = accuracy;
    }

    public double getAccuracy(){
        return accuracy;
    }

}
