package Classification.Classifier;

import Classification.Parameter.BaggingParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class BaggingTest extends ClassifierTest{

    @Test
    public void testTrain() {
        Bagging bagging = new Bagging();
        BaggingParameter baggingParameter = new BaggingParameter(1, 100);
        bagging.train(iris.getInstanceList(), baggingParameter);
        assertEquals(2.67, 100 * bagging.test(iris.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(bupa.getInstanceList(), baggingParameter);
        assertEquals(42.03, 100 * bagging.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(dermatology.getInstanceList(), baggingParameter);
        assertEquals(1.09, 100 * bagging.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(car.getInstanceList(), baggingParameter);
        assertEquals(0.0, 100 * bagging.test(car.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(tictactoe.getInstanceList(), baggingParameter);
        assertEquals(0.0, 100 * bagging.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(nursery.getInstanceList(), baggingParameter);
        assertEquals(0.0, 100 * bagging.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

}