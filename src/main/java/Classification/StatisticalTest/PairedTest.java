package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;

public abstract class PairedTest {
    public abstract StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable;

    /**
     * Compares two classification algorithms based on their performances (accuracy or error rate). The method first
     * checks the null hypothesis mu1 is less than mu2, if the test rejects this null hypothesis with alpha level of confidence, it
     * decides mu1 is larger than mu2. The algorithm then checks the null hypothesis mu1 is larger than mu2, if the test rejects that null
     * hypothesis with alpha level of confidence, if decides mu1 is less than mu2. If none of the two tests are rejected, it can not
     * make a decision about the performances of algorithms.
     * @param classifier1 Performance (error rate or accuracy) results of the first classifier.
     * @param classifier2 Performance (error rate or accuracy) results of the second classifier.
     * @param alpha Alpha level defined for the statistical test.
     * @return 1 if the performance of the first algorithm is larger than the second algorithm, -1 if the performance of
     * the second algorithm is larger than the first algorithm, 0 if they have similar performance.
     * @throws StatisticalTestNotApplicable If the number of experiments do not match, then the function throws
     * StatisticalTestNotApplicable.
     */
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
