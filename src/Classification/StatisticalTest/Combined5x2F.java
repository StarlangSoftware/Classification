package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;
import Math.Distribution;

public class Combined5x2F extends PairedTest{

    private double testStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        if (classifier1.numberOfExperiments() != classifier2.numberOfExperiments()){
            throw new StatisticalTestNotApplicable("Combined 5x2 F test", "In order to apply a paired test, you need to have the same number of experiments in both algorithms.");
        }
        if (classifier1.numberOfExperiments() != 10){
            throw new StatisticalTestNotApplicable("Combined 5x2 F test", "In order to apply a 5x2 test, you need to have 10 experiments.");
        }
        double[] difference;
        difference = new double[classifier1.numberOfExperiments()];
        double numerator = 0;
        for (int i = 0; i < classifier1.numberOfExperiments(); i++){
            difference[i] = classifier1.getErrorRate(i) - classifier2.getErrorRate(i);
            numerator += difference[i] * difference[i];
        }
        double denominator = 0;
        for (int i = 0; i < classifier1.numberOfExperiments() / 2; i++){
            double mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
            double variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) + (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
            denominator += variance;
        }
        denominator *= 2;
        if (denominator == 0){
            throw new StatisticalTestNotApplicable("Combined 5x2 F test", "Variance is 0.");
        }
        return numerator / denominator;
    }

    public StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        double statistic = testStatistic(classifier1, classifier2);
        int degreeOfFreedom1 = classifier1.numberOfExperiments();
        int degreeOfFreedom2 = classifier1.numberOfExperiments() / 2;
        return new StatisticalTestResult(Distribution.fDistribution(statistic, degreeOfFreedom1, degreeOfFreedom2), true);
    }
}
