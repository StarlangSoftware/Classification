package Classification.Model;

import Classification.Instance.Instance;

import java.io.*;

public abstract class Model implements Serializable{

    public abstract String predict(Instance instance);

    public void save(String fileName){
        FileOutputStream outFile;
        ObjectOutputStream outObject;
        try {
            outFile = new FileOutputStream(fileName);
            outObject = new ObjectOutputStream (outFile);
            outObject.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
