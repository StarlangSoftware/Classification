package Classification.StatisticalTest;

public class StatisticalTestResult {
    private final double pValue;
    private final boolean onlyTwoTailed;

    public StatisticalTestResult(double pValue, boolean onlyTwoTailed){
        this.pValue = pValue;
        this.onlyTwoTailed = onlyTwoTailed;
    }

    public StatisticalTestResultType oneTailed(double alpha) throws StatisticalTestNotApplicable {
        if (onlyTwoTailed){
            throw new StatisticalTestNotApplicable("The test", "One tailed option is not available for this test. The distribution is one tailed distribution.");
        }
        if (pValue < alpha){
            return StatisticalTestResultType.REJECT;
        } else {
            return StatisticalTestResultType.FAILED_TO_REJECT;
        }
    }

    public StatisticalTestResultType twoTailed(double alpha){
        if (onlyTwoTailed){
            if (pValue < alpha){
                return StatisticalTestResultType.REJECT;
            } else {
                return StatisticalTestResultType.FAILED_TO_REJECT;
            }
        } else {
            if (pValue < alpha / 2 || pValue > 1 - alpha / 2){
                return StatisticalTestResultType.REJECT;
            } else {
                return StatisticalTestResultType.FAILED_TO_REJECT;
            }
        }
    }

    public double getPValue(){
        return pValue;
    }

}
