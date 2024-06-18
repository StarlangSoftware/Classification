package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;

import java.util.ArrayList;
import java.util.HashMap;

public class OneVersusRest extends EnsembleModel {

    public OneVersusRest(ArrayList<Model> models) {
        super(models);
    }

    @Override
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        ArrayList<InstanceList> instanceLists = new ArrayList<>();
        ArrayList<String> classLabels = trainSet.getDistinctClassLabels();
        for (String classLabel : classLabels) {
            InstanceList instanceList = new InstanceList();
            instanceLists.add(instanceList);
            for (int j = 0; j < trainSet.size(); j++) {
                Instance currentInstance;
                if (trainSet.get(j).getClassLabel().equals(classLabel)) {
                    currentInstance = new Instance(classLabel);
                } else {
                    currentInstance = new Instance("-");
                }
                for (int k = 0; k < trainSet.get(j).attributeSize(); k++) {
                    currentInstance.addAttribute(trainSet.get(j).getAttribute(k));
                }
                instanceList.add(currentInstance);
            }
        }
        for (int i = 0; i < instanceLists.size(); i++) {
            models.get(i).train(instanceLists.get(i), parameters);
        }
    }

    @Override
    public String predict(Instance instance) {
        String bestPrediction = null;
        double probability = Integer.MIN_VALUE;
        HashMap<String, Double> map = this.predictProbability(instance);
        for (String key : map.keySet()) {
            if (map.get(key) > probability) {
                probability = map.get(key);
                bestPrediction = key;
            }
        }
        return bestPrediction;
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        HashMap<String, Double> map = new HashMap<>();
        for (Model model : models) {
            HashMap<String, Double> map2 = model.predictProbability(instance);
            for (String key : map2.keySet()) {
                if (!key.equals("-")) {
                    map.put(key, map2.get(key));
                    break;
                }
            }
        }
        return map;
    }

    @Override
    public void loadModel(String fileName) {

    }
}
