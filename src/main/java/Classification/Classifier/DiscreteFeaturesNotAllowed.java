package Classification.Classifier;

public class DiscreteFeaturesNotAllowed extends Exception{

    public String toString(){
        return "Discrete Features are not allowed for this classifier";
    }
}
