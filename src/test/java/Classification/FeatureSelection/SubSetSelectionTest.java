package Classification.FeatureSelection;

import Classification.Classifier.*;
import Classification.Experiment.Experiment;
import Classification.Experiment.KFoldRun;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.KnnModel;
import Classification.Model.LdaModel;
import Classification.Model.NaiveBayesModel;
import Classification.Parameter.*;
import org.junit.Test;

import static org.junit.Assert.*;

public class SubSetSelectionTest extends ClassifierTest {

    @Test
    public void testSubSetSelectionC45() {
        KFoldRun kFoldRun = new KFoldRun(10);
        SubSetSelection forwardSelection = new ForwardSelection();
        Experiment experiment = new Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris);
        assertEquals(1, forwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection backwardSelection = new BackwardSelection(iris.attributeCount());
        assertEquals(3, backwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection floatingSelection = new FloatingSelection();
        assertEquals(1, floatingSelection.execute(kFoldRun, experiment).size());
    }

    @Test
    public void testSubSetSelectionNaiveBayes() {
        KFoldRun kFoldRun = new KFoldRun(10);
        SubSetSelection forwardSelection = new ForwardSelection();
        Experiment experiment = new Experiment(new NaiveBayesModel(), new Parameter(1), nursery);
        assertEquals(3, forwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection backwardSelection = new BackwardSelection(nursery.attributeCount());
        assertEquals(8, backwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection floatingSelection = new FloatingSelection();
        assertEquals(3, floatingSelection.execute(kFoldRun, experiment).size());
    }

    @Test
    public void testSubSetSelectionLda() {
        KFoldRun kFoldRun = new KFoldRun(10);
        SubSetSelection forwardSelection = new ForwardSelection();
        Experiment experiment = new Experiment(new LdaModel(), new Parameter(1), dermatology);
        assertEquals(11, forwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection backwardSelection = new BackwardSelection(dermatology.attributeCount());
        assertEquals(33, backwardSelection.execute(kFoldRun, experiment).size());
        SubSetSelection floatingSelection = new FloatingSelection();
        assertEquals(11, floatingSelection.execute(kFoldRun, experiment).size());
    }

}