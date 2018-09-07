package Classification.Model.Svm;

public class SolutionInfo {
    private double rho;
    private double[] alpha;

    public SolutionInfo(double rho, double[] alpha) {
        this.rho = rho;
        this.alpha = alpha;
    }

    public double getRho() {
        return rho;
    }

    public double getAlpha(int index){
        return alpha[index];
    }

    public void setAlpha(int index, double value){
        alpha[index] = value;
    }

}
