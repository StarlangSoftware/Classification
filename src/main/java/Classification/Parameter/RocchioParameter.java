package Classification.Parameter;

import Classification.DistanceMetric.DistanceMetric;
import Classification.DistanceMetric.EuclidianDistance;

public class RocchioParameter extends Parameter{

    protected DistanceMetric distanceMetric;

    public RocchioParameter(int seed){
        super(seed);
        distanceMetric = new EuclidianDistance();
    }

    public RocchioParameter(int seed, DistanceMetric distanceMetric){
        super(seed);
        this.distanceMetric = distanceMetric;
    }

    public DistanceMetric getDistanceMetric(){
        return distanceMetric;
    }
}
