package Classification.Model;

import Classification.Classifier.Classifier;
import Classification.DistanceMetric.DistanceMetric;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class KnnModel extends Model implements Serializable{

    private InstanceList data;
    private int k;
    private DistanceMetric distanceMetric;
    
    public KnnModel(InstanceList data, int k, DistanceMetric distanceMetric){
    	this.data = data;
    	this.k = k;
        this.distanceMetric = distanceMetric;
    }
    
    public String predict(Instance instance) {
    	InstanceList nearestNeighbors = nearestNeighbors(instance);
        String predictedClass;
        if (instance instanceof CompositeInstance && nearestNeighbors.size() == 0) {
        	predictedClass = ((CompositeInstance)instance).getPossibleClassLabels().get(0);
        } else {
        	predictedClass = Classifier.getMaximum(nearestNeighbors.getClassLabels());
        }
        return predictedClass;
    }
    
    public InstanceList nearestNeighbors(Instance instance){
    	InstanceList result = new InstanceList();
        ArrayList<KnnInstance> instances = new ArrayList<KnnInstance>();
        ArrayList<String> possibleClassLabels = null;
        if (instance instanceof CompositeInstance) {
        	possibleClassLabels = ((CompositeInstance)instance).getPossibleClassLabels();
        }
        for (int i = 0; i < data.size(); i++){
        	if (!(instance instanceof CompositeInstance) || possibleClassLabels.contains(data.get(i).getClassLabel())) {
        		instances.add(new KnnInstance(data.get(i), distanceMetric.distance(data.get(i), instance)));
        	}
        }
        Collections.sort(instances, new KnnInstanceComparator());
        for (int i = 0; i < Math.min(k, instances.size()); i++){
            result.add(instances.get(i).instance);
        }
        return result;
    }

    private class KnnInstance{
        private double distance;
        private Instance instance;
        private KnnInstance(Instance instance, double distance){
            this.instance = instance;
            this.distance = distance;
        }
        public String toString() {
        	String str = "";
        	str += instance.getClassLabel() + " " + distance;
        	return str;
        }
    }

    private class KnnInstanceComparator implements Comparator<KnnInstance>{
    	public int compare(KnnInstance instance1, KnnInstance instance2) {
            if (instance1.distance < instance2.distance){
                return -1;
            } else {
                if (instance1.distance > instance2.distance){
                    return 1;
                } else {
                    return 0;
                }
            }
        }
    }
}
