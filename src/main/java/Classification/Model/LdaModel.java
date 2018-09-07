package Classification.Model;

import Classification.Instance.Instance;
import Math.*;

import java.io.Serializable;
import java.util.HashMap;

public class LdaModel extends GaussianModel implements Serializable{
    protected HashMap<String, Double> w0;
    protected HashMap<String, Vector> w;

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

    public LdaModel(DiscreteDistribution priorDistribution, HashMap<String, Vector> w, HashMap<String, Double> w0){
        this.priorDistribution = priorDistribution;
        this.w = w;
        this.w0 = w0;
    }

}
