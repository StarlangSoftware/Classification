package Classification.DistanceMetric;

import Classification.Instance.Instance;

public interface DistanceMetric {

    double distance(Instance instance1, Instance instance2);
}
