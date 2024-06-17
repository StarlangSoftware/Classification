package Classification.Classifier;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.QdaModel;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class QdaTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        QdaModel qda = new QdaModel();
        qda.train(iris.getInstanceList(), null);
        qda.saveTxt("models/qda-iris.txt");
        qda.loadModel("models/qda-iris.txt");
        assertEquals(2.00, 100 * qda.test(iris.getInstanceList()).getErrorRate(), 0.01);
        qda.train(bupa.getInstanceList(), null);
        qda.saveTxt("models/qda-bupa.txt");
        qda.loadModel("models/qda-bupa.txt");
        assertEquals(36.52, 100 * qda.test(bupa.getInstanceList()).getErrorRate(), 0.01);
    }

}