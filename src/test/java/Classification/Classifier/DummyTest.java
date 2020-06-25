package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class DummyTest extends ClassifierTest{

    @Test
    public void testTrain() {
        Dummy dummy = new Dummy();
        dummy.train(iris.getInstanceList(), null);
        assertEquals(66.67, 100 * dummy.test(iris.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(bupa.getInstanceList(), null);
        assertEquals(42.03, 100 * dummy.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(dermatology.getInstanceList(), null);
        assertEquals(69.40, 100 * dummy.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(car.getInstanceList(), null);
        assertEquals(29.98, 100 * dummy.test(car.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(tictactoe.getInstanceList(), null);
        assertEquals(34.66, 100 * dummy.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(nursery.getInstanceList(), null);
        assertEquals(66.67, 100 * dummy.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(chess.getInstanceList(), null);
        assertEquals(83.77, 100 * dummy.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }

}