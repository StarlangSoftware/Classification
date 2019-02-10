package Classification.Parameter;

import Classification.DistanceMetric.DistanceMetric;

public class KnnParameter extends KMeansParameter {

    private int k;

    /**
     * Parameters of the K-nearest neighbor classifier.
     *
     * @param seed           Seed is used for random number generation.
     * @param k              Parameter of the K-nearest neighbor algorithm.
     * @param distanceMetric Used to calculate the distance between two instances.
     */
    public KnnParameter(int seed, int k, DistanceMetric distanceMetric) {
        super(seed, distanceMetric);
        this.k = k;
    }

    /**
     * Parameters of the K-nearest neighbor classifier.
     *
     * @param seed           Seed is used for random number generation.
     * @param k              Parameter of the K-nearest neighbor algorithm.
     */
    public KnnParameter(int seed, int k) {
        super(seed);
        this.k = k;
    }

    /**
     * Accessor for the k.
     *
     * @return Value of the k.
     */
    public int getK() {
        return k;
    }

}