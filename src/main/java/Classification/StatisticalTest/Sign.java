package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;

public class Sign extends PairedTest{

    /**
     * Calculates n!.
     * @param n n in n!
     * @return n!.
     */
    private int factorial(int n) {
        int i, result = 1;
        for (i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    /**
     * Calculates m of n that is C(n, m)
     * @param m m in C(m, n)
     * @param n n in C(m, n)
     * @return C(m, n)
     */
    private int binomial(int m, int n) {
        if (n == 0 || m == n)
            return 1;
        else
            return factorial(m) / (factorial(n) * factorial(m - n));
    }

    /**
     * Compares two classification algorithms based on their performances (accuracy or error rate) using sign test.
     * @param classifier1 Performance (error rate or accuracy) results of the first classifier.
     * @param classifier2 Performance (error rate or accuracy) results of the second classifier.
     * @return Statistical test result of the comparison.
     * @throws StatisticalTestNotApplicable If the number of experiments do not match or the number of experiments is
     * not 10, then the function throws StatisticalTestNotApplicable.
     */
    public StatisticalTestResult compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2) throws StatisticalTestNotApplicable {
        if (classifier1.numberOfExperiments() != classifier2.numberOfExperiments()){
            throw new StatisticalTestNotApplicable("Sign test", "In order to apply a paired test, you need to have the same number of experiments in both algorithms.");
        }
        int plus = 0, minus = 0;
        for (int i = 0; i < classifier1.numberOfExperiments(); i++){
            if (classifier1.getErrorRate(i) < classifier2.getErrorRate(i)){
                plus++;
            } else {
                if (classifier1.getErrorRate(i) > classifier2.getErrorRate(i)){
                    minus++;
                }
            }
        }
        int total = plus + minus;
        double pValue = 0.0;
        if (total == 0){
            throw new StatisticalTestNotApplicable("Sign test", "Variance is 0.");
        }
        for (int i = 0; i <= plus; i++){
            pValue += binomial(total, i) / Math.pow(2, total);
        }
        return new StatisticalTestResult(pValue, false);
    }
}
