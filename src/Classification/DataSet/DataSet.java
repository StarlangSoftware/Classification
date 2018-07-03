package Classification.DataSet;

import Classification.Attribute.*;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class DataSet {

    private InstanceList instances;
    private DataDefinition definition;

    public DataSet(){
        definition = null;
        instances = new InstanceList();
    }

    public DataSet(DataDefinition definition){
        this.definition = definition;
        instances = new InstanceList();
    }

    public DataSet(File file){
        int i = 0;
        instances = new InstanceList();
        definition = new DataDefinition();
        try {
            Scanner input = new Scanner(file);
            while (input.hasNext()){
                String instanceText = input.nextLine();
                String[] attributes = instanceText.split(",");
                if (i == 0){
                    for (int j = 0; j < attributes.length - 1; j++){
                        try{
                            Double.parseDouble(attributes[j]);
                            definition.addAttribute(AttributeType.CONTINUOUS);
                        }catch (NumberFormatException e){
                            definition.addAttribute(AttributeType.DISCRETE);
                        }
                    }
                } else {
                    if (attributes.length != definition.attributeCount() + 1){
                        continue;
                    }
                }
                Instance instance = new Instance(attributes[attributes.length - 1]);
                for (int j = 0; j < attributes.length - 1; j++){
                    switch (definition.getAttributeType(j)){
                        case CONTINUOUS:
                            try{
                                instance.addAttribute(new ContinuousAttribute(Double.parseDouble(attributes[j])));
                            } catch (NumberFormatException e){
                            }
                            break;
                        case DISCRETE:
                            instance.addAttribute(new DiscreteAttribute(attributes[j]));
                            break;
                    }
                }
                if (instance.attributeSize() == definition.attributeCount()){
                    instances.add(instance);
                }
                i++;
            }
            input.close();
        } catch (FileNotFoundException e) {
            System.out.println(e.toString());
        }
    }

    public DataSet(DataDefinition definition, String separator, String fileName){
        this.definition = definition;
        instances = new InstanceList(definition, separator, fileName);
    }

    private boolean checkDefinition(Instance instance){
        for (int i = 0; i < instance.attributeSize(); i++){
            if (instance.getAttribute(i) instanceof BinaryAttribute){
                if (definition.getAttributeType(i) != AttributeType.BINARY)
                    return false;
            } else {
                if (instance.getAttribute(i) instanceof DiscreteIndexedAttribute){
                    if (definition.getAttributeType(i) != AttributeType.DISCRETE_INDEXED)
                        return false;
                } else {
                    if (instance.getAttribute(i) instanceof DiscreteAttribute){
                        if (definition.getAttributeType(i) != AttributeType.DISCRETE)
                            return false;
                    } else {
                        if (instance.getAttribute(i) instanceof ContinuousAttribute){
                            if (definition.getAttributeType(i) != AttributeType.CONTINUOUS)
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    private void setDefinition(Instance instance){
        ArrayList<AttributeType> attributeTypes = new ArrayList<>();
        for (int i = 0; i < instance.attributeSize(); i++){
            if (instance.getAttribute(i) instanceof BinaryAttribute){
                attributeTypes.add(AttributeType.BINARY);
            } else {
                if (instance.getAttribute(i) instanceof DiscreteIndexedAttribute){
                    attributeTypes.add(AttributeType.DISCRETE_INDEXED);
                } else {
                    if (instance.getAttribute(i) instanceof DiscreteAttribute){
                        attributeTypes.add(AttributeType.DISCRETE);
                    } else {
                        if (instance.getAttribute(i) instanceof ContinuousAttribute){
                            attributeTypes.add(AttributeType.CONTINUOUS);
                        }
                    }
                }
            }
        }
        definition = new DataDefinition(attributeTypes);
    }

    public int sampleSize(){
        return instances.size();
    }

    public int classCount(){
        return instances.classDistribution().size();
    }

    public int attributeCount(){
        return definition.attributeCount();
    }

    public int discreteAttributeCount(){
        return definition.discreteAttributeCount();
    }

    public int continuousAttributeCount(){
        return definition.continuousAttributeCount();
    }

    public String getClasses(){
        String result;
        ArrayList<String> classLabels = instances.getDistinctClassLabels();
        result = classLabels.get(0);
        for (int i = 1; i < classLabels.size(); i++){
            result = result + ";" + classLabels.get(i);
        }
        return result;
    }

    public String info(String dataSetName){
        String result = "DATASET: " + dataSetName + "\n";
        result = result + "Number of instances: " + sampleSize() + "\n";
        result = result + "Number of distinct class labels: " + classCount() + "\n";
        result = result + "Number of attributes: " + attributeCount() + "\n";
        result = result + "Number of discrete attributes: " + discreteAttributeCount() + "\n";
        result = result + "Number of continuous attributes: " + continuousAttributeCount() + "\n";
        result = result + "Class labels: " + getClasses();
        return result;
    }

    public String toString(String dataSetName){
        return String.format("%20s%15d%15d%20d%15d%15d", dataSetName, sampleSize(), classCount(), attributeCount(), discreteAttributeCount(), continuousAttributeCount());
    }

    public void addInstance(Instance current){
        if (definition == null){
            setDefinition(current);
            instances.add(current);
        } else {
            if (checkDefinition(current)){
                instances.add(current);
            }
        }
    }

    public void addInstanceList(ArrayList<Instance> instanceList){
        for (Instance instance : instanceList){
            addInstance(instance);
        }
    }

    public ArrayList<Instance> getInstances(){
        return instances.getInstances();
    }

    public ArrayList<Instance>[] getClassInstances(){
        return instances.divideIntoClasses().getLists();
    }

    public InstanceList getInstanceList(){
        return instances;
    }

    public DataDefinition getDataDefinition(){
        return definition;
    }

    public DataSet getSubSetOfFeatures(FeatureSubSet featureSubSet){
        DataSet result = new DataSet(definition.getSubSetOfFeatures(featureSubSet));
        for (int i = 0; i < instances.size(); i++){
            result.addInstance(instances.get(i).getSubSetOfFeatures(featureSubSet));
        }
        return result;
    }

    public void writeToFile(String outFileName){
        PrintWriter writer;
        try {
            writer = new PrintWriter(outFileName, "UTF-8");
            for (int i = 0; i < instances.size(); i++){
                writer.write(instances.get(i).toString() + "\n");
            }
            writer.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

}
