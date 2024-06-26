package Classification.Model.Ensemble;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.Model;
import Classification.Parameter.Parameter;

import java.util.ArrayList;
import java.util.HashMap;

public class OneVersusOne extends EnsembleModel {

    public OneVersusOne(ArrayList<Model> models) {
        super(models);
    }

    @Override
    public String predict(Instance instance) {
        HashMap<String, Double> values = predictProbability(instance);
        double max = Integer.MIN_VALUE;
        String best = null;
        for (String key : values.keySet()) {
            if (values.get(key) > max) {
                max = values.get(key);
                best = key;
            }
        }
        return best;
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        HashMap<String, Double> probabilities = new HashMap<>();
        for (Model m : models) {
            HashMap<String, Double> currentProbabilities = m.predictProbability(instance);
            for (String key : currentProbabilities.keySet()) {
                probabilities.put(key, probabilities.getOrDefault(key, 0.0) + ((currentProbabilities.get(key)) / (models.size() - 1)));
            }
        }
        return probabilities;
    }

    @Override
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        ArrayList<String> classLabels = trainSet.getDistinctClassLabels();
        int currentIndex = -1;
        for (int i = 0; i < classLabels.size() - 1; i++) {
            for (int j = i + 1; j < classLabels.size(); j++) {
                currentIndex++;
                InstanceList instanceList = new InstanceList();
                for (int k = 0; k < trainSet.size(); k++) {
                    String label = trainSet.get(k).getClassLabel();
                    if (label.equals(classLabels.get(i)) || label.equals(classLabels.get(j))) {
                        instanceList.add(trainSet.get(k));
                    }
                }
                models.get(currentIndex).train(instanceList, parameters);
            }
        }
    }

    @Override
    public void loadModel(String fileName) {

    }
}
