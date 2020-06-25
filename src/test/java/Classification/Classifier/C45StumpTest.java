package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class C45StumpTest extends ClassifierTest{

    @Test
    public void testTrain() {
        C45Stump c45Stump = new C45Stump();
        c45Stump.train(iris.getInstanceList(), null);
        assertEquals(33.33, 100 * c45Stump.test(iris.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(bupa.getInstanceList(), null);
        assertEquals(42.03, 100 * c45Stump.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(dermatology.getInstanceList(), null);
        assertEquals(49.73, 100 * c45Stump.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(car.getInstanceList(), null);
        assertEquals(29.98, 100 * c45Stump.test(car.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(tictactoe.getInstanceList(), null);
        assertEquals(30.06, 100 * c45Stump.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(nursery.getInstanceList(), null);
        assertEquals(29.03, 100 * c45Stump.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(chess.getInstanceList(), null);
        assertEquals(80.76, 100 * c45Stump.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }
}