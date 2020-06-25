package Classification.Model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import Classification.Classifier.Classifier;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;

public class RandomModel extends Model implements Serializable {
    private ArrayList<String> classLabels;
    private Random random;

    /**
     * A constructor that sets the class labels.
     *
     * @param classLabels An ArrayList of class labels.
     * @param seed Seed of the random function.
     */
    public RandomModel(ArrayList<String> classLabels, int seed) {
        this.classLabels = classLabels;
        this.random = new Random(seed);
    }

    /**
     * The predict method gets an Instance as an input and retrieves the possible class labels as an ArrayList. Then selects a
     * random number as an index and returns the class label at this selected index.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The class label at the randomly selected index.
     */
    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
            int size = possibleClassLabels.size();
            int index = random.nextInt(size);
            return possibleClassLabels.get(index);
        } else {
            int size = classLabels.size();
            int index = random.nextInt(size);
            return classLabels.get(index);
        }
    }
}
