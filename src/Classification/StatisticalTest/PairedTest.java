package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;

public abstract class PairedTest {
    public abstract StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable;

    public int compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2, double alpha) throws StatisticalTestNotApplicable {
        StatisticalTestResult testResult1 = compare(classifier1, classifier2);
        StatisticalTestResult testResult2 = compare(classifier2, classifier1);
        StatisticalTestResultType testResultType1 = testResult1.oneTailed(alpha);
        StatisticalTestResultType testResultType2 = testResult2.oneTailed(alpha);
        if (testResultType1.equals(StatisticalTestResultType.REJECT)){
            return 1;
        } else {
            if (testResultType2.equals(StatisticalTestResultType.REJECT)){
                return -1;
            } else {
                return 0;
            }
        }
    }
}
