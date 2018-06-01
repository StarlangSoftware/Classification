package Classification.Performance;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class ExperimentPerformance implements Comparable{
    private ArrayList<Performance> results;
    private boolean containsDetails = true;
    private boolean classification = true;

    public ExperimentPerformance(){
        results = new ArrayList<Performance>();
    }

    public ExperimentPerformance(String fileName) throws FileNotFoundException {
        results = new ArrayList<>();
        containsDetails = false;
        Scanner input = new Scanner(new File(fileName));
        while (input.hasNext()){
            String performance = input.nextLine();
            results.add(new Performance(Double.parseDouble(performance)));
        }
    }

    public void add(Performance performance){
        if (!(performance instanceof DetailedClassificationPerformance)){
            containsDetails = false;
        }
        if (!(performance instanceof ClassificationPerformance)){
            classification = false;
        }
        results.add(performance);
    }

    public int numberOfExperiments(){
        return results.size();
    }

    public double getErrorRate(int index){
        return results.get(index).getErrorRate();
    }

    public double getAccuracy(int index) throws ClassificationAlgorithmExpectedException {
        if (results.get(index) instanceof ClassificationPerformance){
            return ((ClassificationPerformance) results.get(index)).getAccuracy();
        } else {
            throw new ClassificationAlgorithmExpectedException();
        }
    }

    public Performance meanPerformance(){
        double sumError = 0;
        for (Performance performance :results){
            sumError += performance.getErrorRate();
        }
        return new Performance(sumError / results.size());
    }

    public ClassificationPerformance meanClassificationPerformance(){
        if (results.isEmpty() || !classification){
            return null;
        }
        double sumAccuracy = 0;
        for (Performance performance :results){
            ClassificationPerformance classificationPerformance = (ClassificationPerformance) performance;
            sumAccuracy += classificationPerformance.getAccuracy();
        }
        return new ClassificationPerformance(sumAccuracy / results.size());
    }

    public DetailedClassificationPerformance meanDetailedPerformance(){
        if (results.isEmpty() || !containsDetails){
            return null;
        }
        ConfusionMatrix sum = ((DetailedClassificationPerformance) results.get(0)).getConfusionMatrix();
        for (int i = 1; i < results.size(); i++){
            sum.addConfusionMatrix(((DetailedClassificationPerformance) results.get(i)).getConfusionMatrix());
        }
        return new DetailedClassificationPerformance(sum);
    }

    public Performance standardDeviationPerformance(){
        double sumErrorRate = 0;
        Performance averagePerformance;
        averagePerformance = meanPerformance();
        for (Performance performance :results){
            sumErrorRate += Math.pow(performance.getErrorRate() - averagePerformance.getErrorRate(), 2);
        }
        return new Performance(Math.sqrt(sumErrorRate / (results.size() - 1)));
    }

    public ClassificationPerformance standardDeviationClassificationPerformance(){
        if (results.isEmpty() || !classification){
            return null;
        }
        double sumAccuracy = 0, sumErrorRate = 0;
        ClassificationPerformance averageClassificationPerformance;
        averageClassificationPerformance = meanClassificationPerformance();
        for (Performance performance :results){
            ClassificationPerformance classificationPerformance = (ClassificationPerformance) performance;
            sumAccuracy += Math.pow(classificationPerformance.getAccuracy() - averageClassificationPerformance.getAccuracy(), 2);
            sumErrorRate += Math.pow(classificationPerformance.getErrorRate() - averageClassificationPerformance.getErrorRate(), 2);
        }
        return new ClassificationPerformance(Math.sqrt(sumAccuracy / (results.size() - 1)), Math.sqrt(sumErrorRate / (results.size() - 1)));
    }

    public boolean isBetter(ExperimentPerformance experimentPerformance){
        return compareTo(experimentPerformance) > 0;
    }

    @Override
    public int compareTo(Object o) {
        if (o instanceof ExperimentPerformance){
            double accuracy1, accuracy2;
            accuracy1 = meanClassificationPerformance().getAccuracy();
            accuracy2 = ((ExperimentPerformance) o).meanClassificationPerformance().getAccuracy();
            return Double.compare(accuracy1, accuracy2);
        } else {
            return 0;
        }
    }
}
