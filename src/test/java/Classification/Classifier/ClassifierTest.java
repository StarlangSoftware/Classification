package Classification.Classifier;

import Classification.Attribute.AttributeType;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;

import java.util.ArrayList;

public class ClassifierTest {
    protected DataSet iris, car, chess, bupa, tictactoe, dermatology, nursery;

    @org.junit.Before
    public void setUp() throws Exception {
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 4; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        iris = new DataSet(dataDefinition, ",", "datasets/iris.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        dataDefinition = new DataDefinition(attributeTypes);
        bupa = new DataSet(dataDefinition, ",", "datasets/bupa.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 34; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        dataDefinition = new DataDefinition(attributeTypes);
        dermatology = new DataSet(dataDefinition, ",", "datasets/dermatology.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        dataDefinition = new DataDefinition(attributeTypes);
        car = new DataSet(dataDefinition, ",", "datasets/car.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 9; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        dataDefinition = new DataDefinition(attributeTypes);
        tictactoe = new DataSet(dataDefinition, ",", "datasets/tictactoe.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 8; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        dataDefinition = new DataDefinition(attributeTypes);
        nursery = new DataSet(dataDefinition, ",", "datasets/nursery.data");
        attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            if (i % 2 == 0){
                attributeTypes.add(AttributeType.DISCRETE);
            } else {
                attributeTypes.add(AttributeType.CONTINUOUS);
            }
        }
        dataDefinition = new DataDefinition(attributeTypes);
        chess = new DataSet(dataDefinition, ",", "datasets/chess.data");
    }

}
