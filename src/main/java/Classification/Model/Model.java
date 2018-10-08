package Classification.Model;

import Classification.Instance.Instance;

import java.io.*;

public abstract class Model implements Serializable {

    /**
     * An abstract predict method that takes an {@link Instance} as an input.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The class label as a String.
     */
    public abstract String predict(Instance instance);

    /**
     * The save metohd takes a file name as an input and writes model to that file.
     *
     * @param fileName File name.
     */
    public void save(String fileName) {
        FileOutputStream outFile;
        ObjectOutputStream outObject;
        try {
            outFile = new FileOutputStream(fileName);
            outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
