package Classification.Parameter;

import Classification.DistanceMetric.DistanceMetric;

public class KnnParameter extends RocchioParameter{

    private int k;

    public KnnParameter(int seed, int k, DistanceMetric distanceMetric){
        super(seed, distanceMetric);
        this.k = k;
    }

    public int getK(){
        return k;
    }

}
