package Classification.Model;

import Classification.Instance.Instance;
import Math.*;

import java.io.Serializable;
import java.util.HashMap;

public class LdaModel extends GaussianModel implements Serializable {
    protected HashMap<String, Double> w0;
    protected HashMap<String, Vector> w;

    /**
     * A constructor which sets the priorDistribution, w and w0 according to given inputs.
     *
     * @param priorDistribution {@link DiscreteDistribution} input.
     * @param w                 {@link HashMap} of String and Vectors.
     * @param w0                {@link HashMap} of String and Double.
     */
    public LdaModel(DiscreteDistribution priorDistribution, HashMap<String, Vector> w, HashMap<String, Double> w0) {
        this.priorDistribution = priorDistribution;
        this.w = w;
        this.w0 = w0;
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs. It returns the dot product of given Instance
     * and wi plus w0i.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The dot product of given Instance and wi plus w0i.
     */
    @Override
    protected double calculateMetric(Instance instance, String Ci) {
        double w0i;
        Vector xi, wi;
        xi = instance.toVector();
        wi = w.get(Ci);
        w0i = w0.get(Ci);
        try {
            return wi.dotProduct(xi) + w0i;
        } catch (VectorSizeMismatch vectorSizeMismatch) {
            return Double.MAX_VALUE;
        }
    }

}
