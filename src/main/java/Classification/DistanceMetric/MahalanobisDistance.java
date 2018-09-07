package Classification.DistanceMetric;

import Classification.Instance.Instance;
import Math.*;

import java.io.Serializable;

public class MahalanobisDistance implements DistanceMetric, Serializable {

    private Matrix covarianceInverse;

    /**
     * Constructor for the MahalanobisDistance class. Basically sets the inverse of the covariance matrix.
     * @param covarianceInverse Inverse of the covariance matrix.
     */
    public MahalanobisDistance(Matrix covarianceInverse){
        this.covarianceInverse = covarianceInverse;
    }

    /**
     * Calculates Mahalanobis distance between two instances. (x^(1) - x^(2)) S (x^(1) - x^(2))^T
     * @param instance1 First instance.
     * @param instance2 Second instance.
     * @return Mahalanobis distance between two instances.
     */
    public double distance(Instance instance1, Instance instance2) {
        Vector v1, v2, v3;
        v1 = instance1.toVector();
        v2 = instance2.toVector();
        try {
            v1.subtract(v2);
            v3 = covarianceInverse.multiplyWithVectorFromLeft(v1);
            return v3.dotProduct(v1);
        } catch (MatrixRowMismatch | VectorSizeMismatch matrixRowMismatch) {
            return Integer.MAX_VALUE;
        }
    }
}
