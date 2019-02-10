package Classification.Parameter;

import Classification.DistanceMetric.DistanceMetric;
import Classification.DistanceMetric.EuclidianDistance;

public class KMeansParameter extends Parameter {

    protected DistanceMetric distanceMetric;

    /**
     * Parameters of the Rocchio classifier.
     *
     * @param seed Seed is used for random number generation.
     */
    public KMeansParameter(int seed) {
        super(seed);
        distanceMetric = new EuclidianDistance();
    }

    /**
     * * Parameters of the Rocchio classifier.
     *
     * @param seed           Seed is used for random number generation.
     * @param distanceMetric distance metric used to calculate the distance between two instances.
     */
    public KMeansParameter(int seed, DistanceMetric distanceMetric) {
        super(seed);
        this.distanceMetric = distanceMetric;
    }

    /**
     * Accessor for the distanceMetric.
     *
     * @return The distanceMetric.
     */
    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
    }
}