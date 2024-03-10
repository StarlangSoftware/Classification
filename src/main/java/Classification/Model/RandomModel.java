package Classification.Model;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;

public class RandomModel extends Model implements Serializable {
    private final ArrayList<String> classLabels;
    private final Random random;

    private final int seed;

    /**
     * A constructor that sets the class labels.
     *
     * @param classLabels An ArrayList of class labels.
     * @param seed Seed of the random function.
     */
    public RandomModel(ArrayList<String> classLabels, int seed) {
        this.classLabels = classLabels;
        this.random = new Random(seed);
        this.seed = seed;
    }

    public RandomModel(String fileName){
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            seed = Integer.parseInt(input.readLine());
            random = new Random(seed);
            int size = Integer.parseInt(input.readLine());
            classLabels = new ArrayList<>();
            for (int i = 0; i < size; i++){
                classLabels.add(input.readLine());
            }
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The predict method gets an Instance as an input and retrieves the possible class labels as an ArrayList. Then selects a
     * random number as an index and returns the class label at this selected index.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The class label at the randomly selected index.
     */
    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
            int size = possibleClassLabels.size();
            int index = random.nextInt(size);
            return possibleClassLabels.get(index);
        } else {
            int size = classLabels.size();
            int index = random.nextInt(size);
            return classLabels.get(index);
        }
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        HashMap<String, Double> result = new HashMap<>();
        for (String classLabel : classLabels){
            result.put(classLabel, 1.0 / classLabels.size());
        }
        return result;
    }

    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            output.println(seed);
            output.println(classLabels.size());
            for (String classLabel : classLabels){
                output.println(classLabel);
            }
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
