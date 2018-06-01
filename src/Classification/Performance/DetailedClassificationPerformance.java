package Classification.Performance;

public class DetailedClassificationPerformance extends ClassificationPerformance {
    private ConfusionMatrix confusionMatrix;

    public DetailedClassificationPerformance(ConfusionMatrix confusionMatrix){
        super(confusionMatrix.getAccuracy());
        this.confusionMatrix = confusionMatrix;
    }

    public ConfusionMatrix getConfusionMatrix(){
        return confusionMatrix;
    }

}
