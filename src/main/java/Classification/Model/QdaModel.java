package Classification.Model;

import Classification.Instance.Instance;
import Math.*;

import java.io.Serializable;
import java.util.HashMap;

public class QdaModel extends LdaModel implements Serializable{
    private HashMap<String, Matrix> W;

    protected double calculateMetric(Instance instance, String Ci) {
        double w0i;
        Vector xi, wi;
        Matrix Wi;
        xi = instance.toVector();
        Wi = W.get(Ci);
        wi = w.get(Ci);
        w0i = w0.get(Ci);
        try {
            return Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i;
        } catch (VectorSizeMismatch | MatrixRowMismatch vectorSizeMismatch) {
            return Double.MAX_VALUE;
        }
    }

    public QdaModel(DiscreteDistribution priorDistribution, HashMap<String, Matrix> W, HashMap<String, Vector> w, HashMap<String, Double> w0){
        super(priorDistribution, w, w0);
        this.W = W;
    }
}
