package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;
import Math.Distribution;

public class Pairedt extends PairedTest{

    private double testStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        if (classifier1.numberOfExperiments() != classifier2.numberOfExperiments()){
            throw new StatisticalTestNotApplicable("Paired test", "In order to apply a paired test, you need to have the same number of experiments in both algorithms.");
        }
        double[] difference;
        difference = new double[classifier1.numberOfExperiments()];
        double sum = 0.0;
        for (int i = 0; i < classifier1.numberOfExperiments(); i++){
            difference[i] = classifier1.getErrorRate(i) - classifier2.getErrorRate(i);
            sum += difference[i];
        }
        double mean = sum / classifier1.numberOfExperiments();
        sum = 0.0;
        for (int i = 0; i < classifier1.numberOfExperiments(); i++){
            sum += (difference[i] - mean) * (difference[i] - mean);
        }
        double standardDeviation = Math.sqrt(sum / (classifier1.numberOfExperiments() - 1));
        if (standardDeviation == 0){
            throw new StatisticalTestNotApplicable("Paired t test", "Variance is 0.");
        }
        return Math.sqrt(classifier1.numberOfExperiments()) * mean / standardDeviation;
    }

    public StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        double statistic = testStatistic(classifier1, classifier2);
        int degreeOfFreedom = classifier1.numberOfExperiments() - 1;
        return new StatisticalTestResult(Distribution.tDistribution(statistic, degreeOfFreedom), false);
    }
}
