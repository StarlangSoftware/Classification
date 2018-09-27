package Classification.Instance;

import java.util.ArrayList;
import java.util.Arrays;

import Classification.Attribute.Attribute;

public class CompositeInstance extends Instance {

    private ArrayList<String> possibleClassLabels;

    /**
     * Constructor of {@link CompositeInstance} class which takes a class label as an input. It generates a new composite instance
     * with given class label.
     *
     * @param classLabel Class label of the composite instance.
     */
    public CompositeInstance(String classLabel) {
        super(classLabel);
        this.possibleClassLabels = new ArrayList<String>();
    }

    /**
     * Constructor of {@link CompositeInstance} class which takes a class label and attributes as inputs. It generates
     * a new composite instance with given class label and attributes.
     *
     * @param classLabel Class label of the composite instance.
     * @param attributes Attributes of the composite instance.
     */
    public CompositeInstance(String classLabel, ArrayList<Attribute> attributes) {
        super(classLabel, attributes);
        this.possibleClassLabels = new ArrayList<String>();
    }

    /**
     * Constructor of {@link CompositeInstance} class which takes an {@link java.lang.reflect.Array} of possible labels as
     * input. It generates a new composite instance with given labels.
     *
     * @param possibleLabels Possible labels of the composite instance.
     */
    public CompositeInstance(String[] possibleLabels) {
        this(possibleLabels[0]);
        possibleClassLabels.addAll(Arrays.asList(possibleLabels).subList(1, possibleLabels.length));
    }

    /**
     * Constructor of {@link CompositeInstance} class which takes a class label, attributes and an {@link ArrayList} of
     * possible labels as inputs. It generates a new composite instance with given labels, attributes and possible labels.
     *
     * @param classLabel          Class label of the composite instance.
     * @param attributes          Attributes of the composite instance.
     * @param possibleClassLabels Possible labels of the composite instance.
     */
    public CompositeInstance(String classLabel, ArrayList<Attribute> attributes, ArrayList<String> possibleClassLabels) {
        super(classLabel, attributes);
        this.possibleClassLabels = possibleClassLabels;
    }

    /**
     * Accessor for the possible class labels.
     *
     * @return Possible class labels of the composite instance.
     */
    public ArrayList<String> getPossibleClassLabels() {
        return possibleClassLabels;
    }

    /**
     * Mutator method for possible class labels.
     *
     * @param possibleClassLabels Ner value of possible class labels.
     */
    public void setPossibleClassLabels(ArrayList<String> possibleClassLabels) {
        this.possibleClassLabels = possibleClassLabels;
    }

    /**
     * Converts possible class labels to {@link String}.
     *
     * @return String representation of possible class labels.
     */
    public String toString() {
        String result = super.toString();
        for (String possibleClassLabel : possibleClassLabels) {
            result = result + ";" + possibleClassLabel;
        }
        return result;
    }

}
