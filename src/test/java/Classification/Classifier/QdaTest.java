package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class QdaTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        Qda qda = new Qda();
        qda.train(iris.getInstanceList(), null);
        qda.getModel().saveTxt("models/qda-iris.txt");
        qda.loadModel("models/qda-iris.txt");
        assertEquals(2.00, 100 * qda.test(iris.getInstanceList()).getErrorRate(), 0.01);
        qda.train(bupa.getInstanceList(), null);
        qda.getModel().saveTxt("models/qda-bupa.txt");
        qda.loadModel("models/qda-bupa.txt");
        assertEquals(36.52, 100 * qda.test(bupa.getInstanceList()).getErrorRate(), 0.01);
    }

}