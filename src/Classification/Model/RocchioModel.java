package Classification.Model;

import Classification.DistanceMetric.DistanceMetric;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.Serializable;

public class RocchioModel extends GaussianModel implements Serializable{
    private InstanceList classMeans;
    private DistanceMetric distanceMetric;

    public RocchioModel(DiscreteDistribution priorDistribution, InstanceList classMeans, DistanceMetric distanceMetric){
        this.classMeans = classMeans;
        this.priorDistribution = priorDistribution;
        this.distanceMetric = distanceMetric;
    }

    protected double calculateMetric(Instance instance, String Ci) {
        for (int i = 0; i < classMeans.size(); i++){
            if (classMeans.get(i).getClassLabel().equals(Ci)) {
                return -distanceMetric.distance(instance, classMeans.get(i));
            }
        }
        return -Double.MAX_VALUE;
    }

}
