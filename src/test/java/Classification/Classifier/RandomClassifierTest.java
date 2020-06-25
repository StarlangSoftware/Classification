package Classification.Classifier;

import Classification.Parameter.Parameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class RandomClassifierTest extends ClassifierTest{

    @Test
    public void testTrain() {
        RandomClassifier randomClassifier = new RandomClassifier();
        Parameter parameter = new Parameter(1);
        randomClassifier.train(iris.getInstanceList(), parameter);
        assertEquals(69.33, 100 * randomClassifier.test(iris.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(bupa.getInstanceList(), parameter);
        assertEquals(49.86, 100 * randomClassifier.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(dermatology.getInstanceList(), parameter);
        assertEquals(84.97, 100 * randomClassifier.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(car.getInstanceList(), parameter);
        assertEquals(74.65, 100 * randomClassifier.test(car.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(tictactoe.getInstanceList(), parameter);
        assertEquals(49.16, 100 * randomClassifier.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(nursery.getInstanceList(), parameter);
        assertEquals(79.97, 100 * randomClassifier.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        randomClassifier.train(chess.getInstanceList(), parameter);
        assertEquals(94.45, 100 * randomClassifier.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }

}