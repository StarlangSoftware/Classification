package Classification.Parameter;

import Classification.DistanceMetric.DistanceMetric;

public class RocchioParameter extends Parameter{

    protected DistanceMetric distanceMetric;

    public RocchioParameter(int seed){
        super(seed);
    }

    public RocchioParameter(int seed, DistanceMetric distanceMetric){
        super(seed);
        this.distanceMetric = distanceMetric;
    }

    public DistanceMetric getDistanceMetric(){
        return distanceMetric;
    }
}
