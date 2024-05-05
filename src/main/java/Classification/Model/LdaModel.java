package Classification.Model;

import Classification.Instance.Instance;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
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
     * Default constructor
     */
    public LdaModel(){
    }

    /**
     * Loads w0 and w hash maps from an input file. The number of items in the hash map is given by the parameter size.
     * @param input Input file
     * @param size Number of items in the hash map read.
     * @throws IOException If the file can not be read, it throws IOException.
     */
    protected void loadWandW0(BufferedReader input, int size) throws IOException {
        w0 = new HashMap<>();
        for (int i = 0; i < size; i++){
            String line = input.readLine();
            String[] items = line.split(" ");
            w0.put(items[0], Double.parseDouble(items[1]));
        }
        w = loadVectors(input, size);
    }

    /**
     * Loads a Linear Discriminant Analysis model from an input model file.
     * @param fileName Model file name.
     */
    public LdaModel(String fileName){
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            loadWandW0(input, size);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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

    /**
     * Saves w and w0 hash maps to an output file
     * @param output Output file
     */
    protected void saveWandW0(PrintWriter output){
        for (String c : w0.keySet()){
            output.println(c + " " + w0.get(c));
        }
        saveVectors(output, w);
    }

    /**
     * Saves the Linear discriminant analysis model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveWandW0(output);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
