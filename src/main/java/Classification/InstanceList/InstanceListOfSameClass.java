package Classification.InstanceList;

public class InstanceListOfSameClass extends InstanceList{

    private String classLabel;

    public InstanceListOfSameClass(String classLabel){
        super();
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
