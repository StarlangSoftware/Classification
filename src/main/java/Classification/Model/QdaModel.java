package Classification.Model;

import Classification.Instance.Instance;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

public class QdaModel extends LdaModel implements Serializable {
    private final HashMap<String, Matrix> W;

    /**
     * The constructor which sets the priorDistribution, w w1 and HashMap of String Matrix.
     *
     * @param priorDistribution {@link DiscreteDistribution} input.
     * @param W                 {@link HashMap} of String and Matrix.
     * @param w                 {@link HashMap} of String and Vectors.
     * @param w0                {@link HashMap} of String and Double.
     */
    public QdaModel(DiscreteDistribution priorDistribution, HashMap<String, Matrix> W, HashMap<String, Vector> w, HashMap<String, Double> w0) {
        super(priorDistribution, w, w0);
        this.W = W;
    }

    public QdaModel(String fileName){
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            loadWandW0(input, size);
            W = new HashMap<>();
            for (int i = 0; i < size; i++){
                String c = input.readLine();
                Matrix matrix = loadMatrix(input);
                W.put(c, matrix);
            }
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateMetric method takes an {@link Instance} and a String as inputs. It multiplies Matrix Wi with Vector xi
     * then calculates the dot product of it with xi. Then, again it finds the dot product of wi and xi and returns the summation with w0i.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return The result of Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i.
     */
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

    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveWandW0(output);
            for (String c : W.keySet()){
                Matrix matrix = W.get(c);
                output.println(c);
                saveMatrix(output, matrix);
            }
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
