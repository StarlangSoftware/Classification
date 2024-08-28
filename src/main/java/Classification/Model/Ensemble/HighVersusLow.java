package Classification.Model.Ensemble;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.Model;
import Classification.Parameter.Parameter;

import java.util.ArrayList;
import java.util.HashMap;

public class HighVersusLow extends EnsembleModel {

    private final ArrayList<String> sortedClassLabels;
    private final HashMap<String, Integer> indexMap = new HashMap<>();

    public HighVersusLow(ArrayList<Model> models, ArrayList<String> sortedClassLabels) {
        super(models);
        this.sortedClassLabels = sortedClassLabels;
        setIndexMap();
    }

    private void setIndexMap() {
        indexMap.clear();
        for (int i = 0; i < sortedClassLabels.size(); i++) {
            indexMap.put(sortedClassLabels.get(i), i);
        }
    }

    @Override
    public String predict(Instance instance) {
        HashMap<String, Double> map = this.predictProbability(instance);
        String label = null;
        double best = Integer.MIN_VALUE;
        for (String key : map.keySet()) {
            if (map.get(key) > best) {
                best = map.get(key);
                label = key;
            }
        }
        return label;
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        HashMap<String, Double> probabilities = new HashMap<>();
        for (int i = 0; i < models.size(); i++) {
            Model model = models.get(i);
            HashMap<String, Double> map = model.predictProbability(instance);
            String label = sortedClassLabels.get(i);
            for (String curLabel : sortedClassLabels) {
                if (indexMap.get(curLabel) <= indexMap.get(label)) {
                    probabilities.put(curLabel, probabilities.getOrDefault(curLabel, 0.0) + (map.getOrDefault(label, 0.0) / (models.size() - 1)));
                } else {
                    probabilities.put(curLabel, probabilities.getOrDefault(curLabel, 0.0) + (map.getOrDefault("-", 0.0) / (models.size() - 1)));
                }
            }
        }
        return probabilities;
    }

    @Override
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        setIndexMap();
        for (int i = 0; i < sortedClassLabels.size() - 1; i++) {
            String classLabel = sortedClassLabels.get(i);
            InstanceList instanceList = new InstanceList();
            for (int j = 0; j < trainSet.size(); j++) {
                Instance currentInstance;
                if (indexMap.get(trainSet.get(j).getClassLabel()) <= indexMap.get(classLabel)) {
                    currentInstance = new Instance(classLabel);
                } else {
                    currentInstance = new Instance("-");
                }
                for (int k = 0; k < trainSet.get(j).attributeSize(); k++) {
                    currentInstance.addAttribute(trainSet.get(j).getAttribute(k));
                }
                instanceList.add(currentInstance);
            }
            models.get(i).train(instanceList, parameters);
        }
    }

    @Override
    public void loadModel(String fileName) {

    }
}
