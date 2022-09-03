package Classification.Classifier;

import Classification.Parameter.C45Parameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class C45Test extends ClassifierTest{

    @Test
    public void testTrain() {
        C45 c45 = new C45();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        c45.train(iris.getInstanceList(), c45Parameter);
        assertEquals(4.00, 100 * c45.test(iris.getInstanceList()).getErrorRate(), 0.01);
        c45.train(bupa.getInstanceList(), c45Parameter);
        assertEquals(42.03, 100 * c45.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        c45.train(dermatology.getInstanceList(), c45Parameter);
        assertEquals(4.37, 100 * c45.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        c45.train(car.getInstanceList(), c45Parameter);
        assertEquals(8.16, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        assertEquals(14.61, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }
}