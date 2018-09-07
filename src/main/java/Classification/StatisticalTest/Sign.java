package Classification.StatisticalTest;

import Classification.Performance.ExperimentPerformance;

public class Sign extends PairedTest{

    private int factorial(int n) {
        int i, result = 1;
        for (i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    private int binomial(int m, int n) {
        if (n == 0 || m == n)
            return 1;
        else
            return factorial(m) / (factorial(n) * factorial(m - n));
    }

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
