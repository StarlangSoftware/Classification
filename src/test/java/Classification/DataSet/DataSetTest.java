package Classification.DataSet;

import Classification.Attribute.AttributeType;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class DataSetTest {
    DataSet iris, car, chess, bupa, tictactoe, dermatology, nursery;

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

    @org.junit.Test
    public void testSampleSize() {
        assertEquals(150, iris.sampleSize());
        assertEquals(345, bupa.sampleSize());
        assertEquals(366, dermatology.sampleSize());
        assertEquals(1728, car.sampleSize());
        assertEquals(958, tictactoe.sampleSize());
        assertEquals(12960, nursery.sampleSize());
        assertEquals(28056, chess.sampleSize());
    }

    @org.junit.Test
    public void testClassCount() {
        assertEquals(3, iris.classCount());
        assertEquals(2, bupa.classCount());
        assertEquals(6, dermatology.classCount());
        assertEquals(4, car.classCount());
        assertEquals(2, tictactoe.classCount());
        assertEquals(5, nursery.classCount());
        assertEquals(18, chess.classCount());
    }

    @org.junit.Test
    public void testGetClasses() {
        assertEquals("Iris-setosa;Iris-versicolor;Iris-virginica", iris.getClasses());
        assertEquals("1;2", bupa.getClasses());
        assertEquals("2;1;3;5;4;6", dermatology.getClasses());
        assertEquals("unacc;acc;vgood;good", car.getClasses());
        assertEquals("positive;negative", tictactoe.getClasses());
        assertEquals("recommend;priority;not_recom;very_recom;spec_prior", nursery.getClasses());
        assertEquals("draw;zero;one;two;three;four;five;six;seven;eight;nine;ten;eleven;twelve;thirteen;fourteen;fifteen;sixteen", chess.getClasses());
    }

}