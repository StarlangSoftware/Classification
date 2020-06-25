package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class LdaTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        Lda lda = new Lda();
        lda.train(iris.getInstanceList(), null);
        assertEquals(2.00, 100 * lda.test(iris.getInstanceList()).getErrorRate(), 0.01);
        lda.train(bupa.getInstanceList(), null);
        assertEquals(29.57, 100 * lda.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        lda.train(dermatology.getInstanceList(), null);
        assertEquals(1.91, 100 * lda.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}