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
	
    public RandomModel(ArrayList<String> classLabels) {
        this.classLabels = classLabels;
    }
    
    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance)instance).getPossibleClassLabels();
            int size = possibleClassLabels.size();
            int index = new Random().nextInt(size);
            return possibleClassLabels.get(index);
        } else {
            int size = classLabels.size();
            int index = new Random().nextInt(size);
            return classLabels.get(index);
        }
    }
}
