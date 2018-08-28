package Classification.Classifier;

public class DiscreteFeaturesNotAllowed extends Exception {

    /**
     * @return Discrete Features are not allowed for the classifier.
     */
    public String toString() {
        return "Discrete Features are not allowed for this classifier";
    }
}
