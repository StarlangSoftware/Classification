package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class NaiveBayesTest extends ClassifierTest{

    @Test
    public void testTrain() {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.train(iris.getInstanceList(), null);
        assertEquals(5.33, 100 * naiveBayes.test(iris.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(bupa.getInstanceList(), null);
        assertEquals(38.55, 100 * naiveBayes.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(dermatology.getInstanceList(), null);
        assertEquals(9.56, 100 * naiveBayes.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(car.getInstanceList(), null);
        assertEquals(12.91, 100 * naiveBayes.test(car.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(tictactoe.getInstanceList(), null);
        assertEquals(30.17, 100 * naiveBayes.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(nursery.getInstanceList(), null);
        assertEquals(9.70, 100 * naiveBayes.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

}