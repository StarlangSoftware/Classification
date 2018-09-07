package Classification.StatisticalTest;

public class StatisticalTestNotApplicable extends Exception{
    private String test;
    private String reason;

    public StatisticalTestNotApplicable(String test, String reason){
        this.test = test;
        this.reason = reason;
    }

    public String toString(){
        return test + " is not applicable. " + reason;
    }

}
