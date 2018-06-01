package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;
import Math.Distribution;

public class Combined5x2t extends PairedTest{

    private double testStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        if (classifier1.numberOfExperiments() != classifier2.numberOfExperiments()){
            throw new StatisticalTestNotApplicable("Combined 5x2 t test", "In order to apply a paired test, you need to have the same number of experiments in both algorithms.");
        }
        if (classifier1.numberOfExperiments() != 10){
            throw new StatisticalTestNotApplicable("Combined 5x2 t test", "In order to apply a 5x2 test, you need to have 10 experiments.");
        }
        double[] difference;
        difference = new double[classifier1.numberOfExperiments()];
        for (int i = 0; i < classifier1.numberOfExperiments(); i++){
            difference[i] = classifier1.getErrorRate(i) - classifier2.getErrorRate(i);
        }
        double denominator = 0;
        double numerator = 0;
        for (int i = 0; i < classifier1.numberOfExperiments() / 2; i++){
            double mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
            numerator += mean;
            double variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) + (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
            denominator += variance;
        }
        numerator = Math.sqrt(10) * numerator / 5;
        denominator = Math.sqrt(denominator / 5);
        if (denominator == 0){
            throw new StatisticalTestNotApplicable("Combined 5x2 t test", "Variance is 0.");
        }
        return numerator / denominator;
    }

    public StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        double statistic = testStatistic(classifier1, classifier2);
        int degreeOfFreedom = classifier1.numberOfExperiments() / 2;
        return new StatisticalTestResult(Distribution.tDistribution(statistic, degreeOfFreedom), false);
    }

}
