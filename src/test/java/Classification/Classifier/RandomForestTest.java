package Classification.Classifier;

import Classification.Parameter.RandomForestParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class RandomForestTest extends ClassifierTest{

    @Test
    public void testTrain() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(iris.getInstanceList(), randomForestParameter);
        assertEquals(2.67, 100 * randomForest.test(iris.getInstanceList()).getErrorRate(), 0.01);
        randomForest.train(bupa.getInstanceList(), randomForestParameter);
        assertEquals(42.03, 100 * randomForest.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        randomForest.train(dermatology.getInstanceList(), randomForestParameter);
        assertEquals(1.09, 100 * randomForest.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        randomForest.train(car.getInstanceList(), randomForestParameter);
        assertEquals(0.0, 100 * randomForest.test(car.getInstanceList()).getErrorRate(), 0.01);
        randomForest.train(tictactoe.getInstanceList(), randomForestParameter);
        assertEquals(0.0, 100 * randomForest.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

}