package Classification.Instance;

import java.util.ArrayList;
import java.util.Arrays;

import Classification.Attribute.Attribute;

public class CompositeInstance extends Instance {

    private ArrayList<String> possibleClassLabels;

    public CompositeInstance(String classLabel) {
        super(classLabel);
        this.possibleClassLabels = new ArrayList<String>();
    }

    public CompositeInstance(String classLabel, ArrayList<Attribute> attributes) {
        super(classLabel, attributes);
        this.possibleClassLabels = new ArrayList<String>();
    }

    public CompositeInstance(String[] possibleLabels){
        this(possibleLabels[0]);
        possibleClassLabels.addAll(Arrays.asList(possibleLabels).subList(1, possibleLabels.length));
    }

    public CompositeInstance(String classLabel, ArrayList<Attribute> attributes, ArrayList<String> possibleClassLabels) {
        super(classLabel, attributes);
        this.possibleClassLabels = possibleClassLabels;
    }


    public ArrayList<String> getPossibleClassLabels() {
        return possibleClassLabels;
    }

    public void setPossibleClassLabels(ArrayList<String> possibleClassLabels) {
        this.possibleClassLabels = possibleClassLabels;
    }

    public String toString(){
        String result = super.toString();
        for (String possibleClassLabel:possibleClassLabels){
            result = result + ";" + possibleClassLabel;
        }
        return result;
    }

}
